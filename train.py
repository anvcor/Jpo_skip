"""J-PlatPat 验证码识别模型训练（小 CNN + 4 个分类头）

双击 train.bat 即可（用全部已标注的图训练，导出给标注工具 / 插件用）；也可以：
  python train.py                 # 默认读 <下载目录>\\jpp_captcha\\labels.csv
  python train.py --cv 5          # 5 折交叉验证 = cross_validate.bat，看真实准确率、和 ddddocr 公平比较
  python train.py --val 0.15      # 留 15% 做验证
  python train.py --epochs 300    # 数据多了可以多训几轮
  python train.py --no-auto       # 不用“批量采用”(auto) 的标注，只用插件验证 + 人工确认的

输出：
  models/jpp_cnn.onnx + jpp_cnn.json   标注工具会自动识别，可以和 ddddocr 并排对比
  models/jpp_cnn.pt                    PyTorch 权重（继续训练用）
  models/jpp_cnn_report.txt            验证集结果和认错的图
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from label_tool import default_data_dir

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / 'models'
H, W = 48, 240           # 原图 60×300，等比缩小
CROP = 1                 # 原图四周有 1 像素黑框，先裁掉（不然数据增强时会被拉成黑色楔形）
LENGTH, CLASSES = 4, 10
MEAN, STD = 0.5, 0.5
NAME = 'jpp_cnn'


# ---------------- 数据 ----------------

def preprocess(im: Image.Image) -> Image.Image:
    """灰度图 → 裁黑框 → 缩放。识别端（标注工具 / 插件）必须做同样的处理，参数写在 json 里。"""
    if CROP:
        im = im.crop((CROP, CROP, im.width - CROP, im.height - CROP))
    return im.resize((W, H), Image.BILINEAR)


def load_dataset(root: Path, use_auto: bool):
    rows = list(csv.DictReader((root / 'labels.csv').open(encoding='utf-8')))
    ok_sources = {'ok', 'manual'} | ({'auto'} if use_auto else set())
    seen, xs, ys, files, sources, conflicts, dups = {}, [], [], [], [], 0, 0
    human = {'ok', 'manual'}
    for r in rows:
        if r['source'] not in ok_sources or not (len(r['label']) == LENGTH and r['label'].isdigit()):
            continue
        p = root / r['file']
        if not p.exists():
            continue
        im = Image.open(p).convert('L')
        h = hashlib.md5(np.asarray(im).tobytes()).hexdigest()
        if h in seen:                      # 完全相同的图只留一张；人工确认的优先于自动采用的
            i = seen[h]
            dups += 1
            conflicts += ''.join(map(str, ys[i])) != r['label']
            if r['source'] in human and sources[i] not in human:
                ys[i], files[i], sources[i] = [int(c) for c in r['label']], r['file'], r['source']
            continue
        seen[h] = len(xs)
        xs.append(np.asarray(preprocess(im), dtype=np.uint8))
        ys.append([int(c) for c in r['label']])
        files.append(r['file'])
        sources.append(r['source'])
    if dups:
        print(f'去掉了 {dups} 张重复图片' + (f'（其中 {conflicts} 组标注不一致，已优先保留人工确认的）' if conflicts else ''))
    return np.stack(xs), np.array(ys, dtype=np.int64), files, sources


def auto_epochs(n_train, batch):
    """数据少时多训几轮，数据多时少训几轮，让总训练步数保持在 3000 步左右。"""
    return int(min(200, max(60, round(3000 * batch / max(1, n_train)))))


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (N,1,H,W)，值域 0~1。随机仿射 + 亮度对比度 + 噪声，全部在 GPU 上做。"""
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, device=dev) - 0.5) * 2 * np.deg2rad(4)       # ±4°
    sc = 1 + (torch.rand(n, device=dev) - 0.5) * 0.12                 # ±6%
    tx = (torch.rand(n, device=dev) - 0.5) * 2 * 0.08                 # 水平 ±8%
    ty = (torch.rand(n, device=dev) - 0.5) * 2 * 0.08                 # 垂直 ±8%
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin * H / W, tx], 1),
                         torch.stack([sin * W / H, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # 用边缘像素填充，避免引入黑边
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4         # 对比度 ±20%
    b = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.2              # 亮度 ±10%
    x = (x - 0.5) * c + 0.5 + b
    x = x + torch.randn_like(x) * 0.03 * torch.rand(n, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


# ---------------- 模型 ----------------

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CaptchaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 96), block(96, 128))  # → 128×3×15
        self.pool = nn.AdaptiveAvgPool2d((1, 15))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 15, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(256, LENGTH * CLASSES))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).view(-1, LENGTH, CLASSES)


class ExportWrapper(nn.Module):
    """导出时把输出转成概率，方便标注工具直接显示置信度。"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ---------------- 训练 ----------------

def predict(net, x, batch=256):
    """x: 已归一化的 (N,1,H,W)。返回 (预测 N×4, 整串置信度 N)。"""
    net.eval()
    preds, confs = [], []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            prob = torch.softmax(net(x[s:s + batch]), -1)
            preds.append(prob.argmax(-1))
            confs.append(prob.max(-1).values.prod(-1))
    return torch.cat(preds).cpu().numpy(), torch.cat(confs).cpu().numpy()


def to_tensor(a, dev):
    """uint8 (N,H,W) → 归一化后的 (N,1,H,W)。"""
    return (torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(1) / 255.0 - MEAN) / STD


def fit(X, Y, args, dev, log=print, X_val=None, Y_val=None):
    """训练一个模型，返回最后一轮的权重（OneCycle 结束时学习率很低，最后一轮最稳定）。"""
    xt, yt = to_tensor(X, dev), torch.tensor(Y, device=dev)
    xv = to_tensor(X_val, dev) if X_val is not None else None
    net = CaptchaNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    steps = args.epochs * max(1, (len(X) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for s in range(0, len(X), args.batch):
            b = perm[s:s + args.batch]
            x = (augment(xt[b] * STD + MEAN) - MEAN) / STD   # augment 在 0~1 空间里做
            loss = F.cross_entropy(net(x).reshape(-1, CLASSES), yt[b].reshape(-1), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if log and (ep % 20 == 0 or ep == 1):
            msg = f'  第 {ep:3d} 轮  loss {tot / len(X):.3f}'
            if xv is not None:
                p, _ = predict(net, xv)
                msg += f'  验证集整串 {(p == Y_val).all(1).mean():6.1%}  单字 {(p == Y_val).mean():6.1%}'
            log(msg)
    return net


def export_onnx(net, meta):
    net_cpu = CaptchaNet()
    net_cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    net_cpu.eval()
    torch.save(net_cpu.state_dict(), MODELS_DIR / f'{NAME}.pt')
    onnx_path = MODELS_DIR / f'{NAME}.onnx'
    torch.onnx.export(ExportWrapper(net_cpu), torch.zeros(1, 1, H, W), str(onnx_path), input_names=['image'],
                      output_names=['prob'], dynamic_axes={'image': {0: 'n'}, 'prob': {0: 'n'}}, opset_version=13,
                      dynamo=False)
    (MODELS_DIR / f'{NAME}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    # 同时导出给 Chrome 插件用的纯 JS 权重（extension/model/）
    import export_web
    export_web.export(net_cpu, {k: meta[k] for k in ('trained_at', 'samples') if k in meta})
    return onnx_path


def onnx_check(onnx_path, X, Y):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    xo = ((X.astype(np.float32) / 255.0 - MEAN) / STD)[:, None]
    return (sess.run(None, {'image': xo})[0].argmax(-1) == Y).all(1).mean()


def s4(a):
    return ''.join(map(str, a))


def run_train(args, root, X, Y, files, dev):
    """普通训练：留出一部分做验证，导出模型给标注工具 / 插件用。"""
    n = len(X)
    idx = np.random.permutation(n)
    n_val = 0 if args.val <= 0 else max(30, int(n * args.val))
    vi, ti = idx[:n_val], idx[n_val:]
    print(f'可用样本 {n} 张 → 训练 {len(ti)} / 验证 {len(vi)}')
    t0 = time.time()
    net = fit(X[ti], Y[ti], args, dev, X_val=X[vi] if n_val else None, Y_val=Y[vi] if n_val else None)
    meta = {'type': 'multihead', 'height': H, 'width': W, 'channels': 1, 'crop': CROP, 'mean': MEAN, 'std': STD,
            'charset': '0123456789', 'length': LENGTH, 'samples': int(len(ti)),
            'trained_at': time.strftime('%Y-%m-%d %H:%M')}
    lines = [f'训练时间 {meta["trained_at"]}  训练 {len(ti)} 张 / 验证 {len(vi)} 张  用时 {time.time() - t0:.0f} 秒']
    if n_val:
        pred, conf = predict(net, to_tensor(X[vi], dev))
        acc = (pred == Y[vi]).all(1).mean()
        meta['val_acc'] = round(float(acc), 4)
        onnx_path = export_onnx(net, meta)
        lines.append(f'验证集整串准确率 {acc:.1%}  单字 {(pred == Y[vi]).mean():.1%}'
                     f'  ONNX 复核 {onnx_check(onnx_path, X[vi], Y[vi]):.1%}')
        wrong = [(files[vi[i]], s4(Y[vi][i]), s4(pred[i]), conf[i])
                 for i in range(len(vi)) if (pred[i] != Y[vi][i]).any()]
        lines += ['', f'验证集认错 {len(wrong)} 张（文件 / 标注 / 预测 / 置信度）：']
        lines += [f'  {f}  {t}  {p}  {c:.3f}' for f, t, p, c in wrong]
    else:
        onnx_path = export_onnx(net, meta)
        lines.append(f'全部 {len(ti)} 张都用来训练了（训练集上 ONNX 复核 {onnx_check(onnx_path, X[ti], Y[ti]):.1%}，'
                     '这个数字不代表真实准确率）')
    lines += ['', f'模型 → {onnx_path}', f'插件用的模型 → {HERE / "extension" / "model"}（到 chrome://extensions 重新加载插件即可生效）',
              '真实准确率、以及和 ddddocr 的公平比较，请运行 cross_validate.bat（交叉验证）。']
    (MODELS_DIR / f'{NAME}_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))


def run_cv(args, root, X, Y, files, sources, dev):
    """K 折交叉验证：每张图都由“没见过它”的模型来预测，结果可以和 ddddocr 公平比较。"""
    k, n = args.cv, len(X)
    folds = np.array_split(np.random.permutation(n), k)
    oof_pred = np.zeros((n, LENGTH), dtype=np.int64)
    oof_conf = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i, vi in enumerate(folds, 1):
        ti = np.setdiff1d(np.arange(n), vi)
        print(f'第 {i}/{k} 折：用 {len(ti)} 张训练（{args.epochs} 轮），预测另外 {len(vi)} 张')
        net = fit(X[ti], Y[ti], args, dev, log=None)
        p, c = predict(net, to_tensor(X[vi], dev))
        oof_pred[vi], oof_conf[vi] = p, c
        print(f'  这一折整串准确率 {(p == Y[vi]).all(1).mean():.1%}    （已用 {time.time() - t0:.0f} 秒）')

    # 写成标注工具能直接显示的预测缓存（模型名 jpp_cnn_cv，只读）
    pred_dir = root / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    cv_preds = {files[i]: [s4(oof_pred[i]), float(oof_conf[i])] for i in range(n)}
    (pred_dir / f'{NAME}_cv.json').write_text(json.dumps(cv_preds, ensure_ascii=False), encoding='utf-8')

    truth = {files[i]: s4(Y[i]) for i in range(n)}
    human = {files[i] for i in range(n) if sources[i] in ('ok', 'manual')}
    ours = {f: cv_preds[f][0] == t for f, t in truth.items()}
    others = {}
    for pf in sorted(pred_dir.glob('*.json')):
        if not pf.stem.startswith(NAME):
            others[pf.stem] = {f: v[0] for f, v in json.loads(pf.read_text(encoding='utf-8')).items()}

    def acc_on(subset):
        rows = [f'  {"jpp_cnn（交叉验证）":<18} {np.mean([ours[f] for f in subset]):.1%}']
        for name, pr in others.items():
            hit = [pr[f] == truth[f] for f in subset if f in pr]
            if hit:
                rows.append(f'  {name:<18} {np.mean(hit):.1%}（{len(hit)} 张）')
        return rows

    lines = [f'{k} 折交叉验证  样本 {n} 张（人工确认 {len(human)}，自动采用 {n - len(human)}）'
             f'  每折训练 {args.epochs} 轮  用时 {time.time() - t0:.0f} 秒', '']
    if human:
        lines += [f'【公平对比】只看人工确认的 {len(human)} 张（自动采用的图本来就是 ddddocr 和 jpp_cnn 一致才采用的，'
                  '放进来会让两者都虚高）：'] + acc_on(sorted(human))
    lines += ['', f'全部 {n} 张（含自动采用，仅供参考）：'] + acc_on(list(truth))

    if 'ddddocr' in others and human:
        dd = others['ddddocr']
        hs = sorted(human)
        both_ok = sum(ours[f] and dd.get(f) == truth[f] for f in hs)
        only_ours = sum(ours[f] and dd.get(f) != truth[f] for f in hs)
        only_dd = sum((not ours[f]) and dd.get(f) == truth[f] for f in hs)
        agree = [f for f in hs if cv_preds[f][0] == dd.get(f)]
        agree_ok = sum(cv_preds[f][0] == truth[f] for f in agree)
        lines += ['', f'和 ddddocr 对照（人工确认的 {len(hs)} 张）：',
                  f'  两个都对 {both_ok}   只有 jpp_cnn 对 {only_ours}   只有 ddddocr 对 {only_dd}'
                  f'   都错 {len(hs) - both_ok - only_ours - only_dd}',
                  f'  两个模型结果一致的 {len(agree)} 张里，{agree_ok} 张和标注相同（{agree_ok / max(1, len(agree)):.1%}）']
        suspect = [f for f in agree if cv_preds[f][0] != truth[f]]
        if suspect:
            lines += ['', f'⚠ 两个模型一致、但和标注不同的 {len(suspect)} 张（可能是标注写错了，建议复查）：']
            lines += [f'  {f}  标注 {truth[f]}  两个模型都认为是 {cv_preds[f][0]}' for f in suspect]

    # 置信度门槛：插件里“没把握就换一张”用
    conf = np.array([cv_preds[f][1] for f in truth])
    right = np.array([ours[f] for f in truth])
    lines += ['', 'jpp_cnn 置信度门槛（低于门槛就不填、换一张）：']
    for th in (0.5, 0.7, 0.8, 0.9, 0.95):
        m = conf >= th
        if m.any():
            lines.append(f'  ≥ {th:<4}：覆盖 {m.mean():6.1%}，其中准确率 {right[m].mean():.2%}')

    has_dd = 'ddddocr' in others
    wrong = [f for f in truth if not ours[f]]
    lines += ['', f'jpp_cnn 认错的 {len(wrong)} 张（文件 / 标注 / jpp_cnn / 置信度' + (' / ddddocr' if has_dd else '') + ' / 来源）：']
    src = dict(zip(files, sources))
    for f in wrong:
        extra = f'  {others["ddddocr"].get(f, "?")}' if has_dd else ''
        lines.append(f'  {f}  {truth[f]}  {cv_preds[f][0]}  {cv_preds[f][1]:.3f}{extra}  {src[f]}')
    lines += ['', '标注工具里的 jpp_cnn_cv 模型（只读）就是这些交叉验证的预测，可以拿来复查。']
    (MODELS_DIR / f'{NAME}_cv_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description='训练验证码识别模型')
    ap.add_argument('--dir', type=Path, default=None)
    ap.add_argument('--epochs', type=int, default=None, help='训练轮数；默认按数据量自动定（总步数约 3000）')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val', type=float, default=0.0, help='留出做验证的比例；默认 0 = 全部用来训练（准确率看交叉验证）')
    ap.add_argument('--cv', type=int, default=0, metavar='K', help='做 K 折交叉验证（不导出模型）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-auto', action='store_true', help='不用 auto（批量采用）的标注')
    args = ap.parse_args()

    root = args.dir or default_data_dir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, Y, files, sources = load_dataset(root, use_auto=not args.no_auto)
    if len(X) < 50:
        raise SystemExit(f'可用标注只有 {len(X)} 张，太少了，至少先标 50 张。')
    if args.epochs is None:
        n_train = len(X) * (args.cv - 1) // args.cv if args.cv else int(len(X) * (1 - args.val))
        args.epochs = auto_epochs(n_train, args.batch)
    print(f'数据：{root}    样本 {len(X)} 张    设备：{dev}    训练 {args.epochs} 轮'
          f'    参数量 {sum(p.numel() for p in CaptchaNet().parameters()) / 1e6:.2f}M')
    MODELS_DIR.mkdir(exist_ok=True)
    if args.cv:
        run_cv(args, root, X, Y, files, sources, dev)
    else:
        run_train(args, root, X, Y, files, dev)


if __name__ == '__main__':
    main()

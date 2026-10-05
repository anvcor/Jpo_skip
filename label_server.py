"""J-PlatPat 验证码标注 —— 本地网页版后端（只用标准库 + PIL/numpy，可选 ddddocr/onnxruntime）

启动：双击 start_label.bat，或 python label_server.py [--dir 数据目录] [--port 8765]

模型：
  - ddddocr          内置（已安装 ddddocr 时可用）
  - ppllocr          内置（已安装 ppllocr 时可用；用 pip install --no-deps ppllocr 安装，避免它把 numpy 升到 2.x）
  - models/*.onnx    自己训练的模型，旁边放同名 .json 说明输入尺寸和解码方式，例如
      {"type": "multihead", "height": 64, "width": 192, "channels": 1, "crop": 0,
       "mean": 0.0, "std": 1.0, "charset": "0123456789", "length": 4}
    type 也可以是 "ctc"（另加 "blank": 0）
预测结果缓存在 <数据目录>/predictions/<模型名>.json，标注写入 <数据目录>/labels.csv（与 label_tool.py 通用）。
"""
import argparse
import json
import mimetypes
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from label_tool import LEADING_DIGITS, default_data_dir, import_ok, list_images, load_labels, rel, save_labels

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / 'label_web'
MODELS_DIR = HERE / 'models'
FOLDERS = ('ok', 'unlabeled', 'wrong')
THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995)


def conf_threshold(name: str, own_model: bool) -> float:
    """界面上“高置信”用的门槛。自己训练的模型训练时用了 label smoothing，整串置信度通常只有 0.8~0.9，用 0.7；
    ddddocr 之类的现成模型置信度普遍接近 1，用 0.99。"""
    return 0.7 if own_model or name.endswith('_cv') else 0.99


# ---------------- 模型 ----------------

class DdddocrModel:
    kind = 'ddddocr'

    def __init__(self):
        import ddddocr
        self.ocr = ddddocr.DdddOcr(show_ad=False)
        self.ocr.set_ranges('0123456789')  # 1.6.1 的 set_ranges(0) 会返回空串

    def predict(self, path: Path):
        import io
        buf = io.BytesIO()
        Image.open(path).convert('RGB').save(buf, 'PNG')
        r = self.ocr.classification(buf.getvalue(), probability=True)
        return r['text'], float(r['confidence'])


class PpllocrModel:
    """ppllocr（YOLOv11 逐字检测）。置信度取各字符中最低的那个，和 ddddocr 的整串置信度刻度不同，不能直接比大小。"""
    kind = 'ppllocr'
    # 字符集含字母，验证码只有数字：把形近字母映射成数字
    LOOKALIKE = str.maketrans({'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1',
                               'Z': '2', 'z': '2', 'S': '5', 's': '5', 'G': '6', 'b': '6', 'B': '8', 'g': '9', 'q': '9'})

    def __init__(self):
        import ppllocr
        self.ocr = ppllocr.OCR()

    def predict(self, path: Path):
        import io
        buf = io.BytesIO()
        Image.open(path).convert('RGB').save(buf, 'PNG')
        _, chars = self.ocr.classification_box(buf.getvalue())
        chars = [c for c in chars if c['char'].translate(self.LOOKALIKE).isdigit()]
        text = ''.join(c['char'] for c in chars).translate(self.LOOKALIKE)
        return text, float(min((c['conf'] for c in chars), default=0.0))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _as_probs(x, axis=-1):
    s = x.sum(axis=axis)
    return x if (x.min() >= 0 and np.allclose(s, 1, atol=1e-3)) else _softmax(x, axis)


class OnnxModel:
    kind = 'onnx'

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        meta_path = onnx_path.with_suffix('.json')
        self.meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        m = self.meta
        self.type = m.get('type', 'multihead')
        self.h, self.w = int(m.get('height', 64)), int(m.get('width', 192))
        self.channels = int(m.get('channels', 1))
        self.crop = int(m.get('crop', 0))  # 缩放前四边各裁掉的像素（去黑框）
        self.mean, self.std = float(m.get('mean', 0.0)), float(m.get('std', 1.0))
        self.charset = m.get('charset', '0123456789')
        self.length = int(m.get('length', 4))
        self.blank = int(m.get('blank', 0))
        self.sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def _prep(self, path):
        im = Image.open(path).convert('L' if self.channels == 1 else 'RGB')
        if self.crop:
            c = self.crop
            im = im.crop((c, c, im.width - c, im.height - c))
        im = im.resize((self.w, self.h), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = x[None, None] if self.channels == 1 else x.transpose(2, 0, 1)[None]
        return x.astype(np.float32)

    def predict(self, path: Path):
        outs = self.sess.run(None, {self.input_name: self._prep(path)})
        if self.type == 'ctc':
            y = np.squeeze(outs[0])                       # (T, C)
            if y.shape[0] == len(self.charset) + 1 and y.shape[1] != len(self.charset) + 1:
                y = y.T
            p = _as_probs(y)
            best = p.argmax(-1)
            text, confs, prev = '', [], -1
            for t, k in enumerate(best):
                if k != prev and k != self.blank:
                    text += self.charset[k - 1 if self.blank == 0 else k]
                    confs.append(p[t, k])
                prev = k
            return text, float(np.prod(confs)) if confs else 0.0
        # multihead：一个 (1, L, C) 输出，或 L 个 (1, C) 输出
        y = np.stack([np.squeeze(o) for o in outs]) if len(outs) == self.length else np.squeeze(outs[0])
        y = y.reshape(self.length, -1)
        p = _as_probs(y)
        idx = p.argmax(-1)
        return ''.join(self.charset[i] for i in idx), float(np.prod(p.max(-1)))


def discover_models():
    """返回 {名称: (工厂函数, 模型文件或 None)}，模型在第一次用到时才加载。"""
    found = {}
    try:
        import ddddocr  # noqa: F401
        found['ddddocr'] = (DdddocrModel, None)
    except ImportError:
        pass
    try:
        import ppllocr  # noqa: F401
        found['ppllocr'] = (PpllocrModel, None)
    except ImportError:
        pass
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob('*.onnx')):
            found[p.stem] = ((lambda p=p: OnnxModel(p)), p)
    return found


def _mtime(p: Path) -> str:
    try:
        return str(p.stat().st_mtime_ns)
    except OSError:
        return ''


# ---------------- 数据与状态 ----------------

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.labels_path = root / 'labels.csv'
        self.pred_dir = root / 'predictions'
        self.pred_dir.mkdir(exist_ok=True)
        legacy = root / 'predictions.json'                     # label_tool.py 的旧缓存
        if legacy.exists() and not (self.pred_dir / 'ddddocr.json').exists():
            legacy.replace(self.pred_dir / 'ddddocr.json')
        self.lock = threading.RLock()
        self.labels = load_labels(self.labels_path)
        self.factories = {}   # 可运行的模型：名称 → 工厂
        self.model_src = {}   # 名称 → 模型文件（ONNX），用来发现“重新训练过”
        self.readonly = {}    # 只有预测缓存、没有模型的（如交叉验证结果）：名称 → 缓存文件 mtime
        self.models = {}
        self.preds = {}
        self.job = {'running': False, 'model': None, 'done': 0, 'total': 0, 'error': ''}
        self.refresh_models()
        self.sync()

    def _pred_file(self, name):
        return self.pred_dir / f'{name}.json'

    def _load_preds(self, name):
        p = self._pred_file(name)
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

    def _save_preds(self, name):
        self._pred_file(name).write_text(json.dumps(self.preds[name], ensure_ascii=False), encoding='utf-8')
        src = self.model_src.get(name)
        if src:  # 记下是用哪个版本的模型预测的
            self._pred_file(name).with_suffix('.model').write_text(_mtime(src), encoding='utf-8')

    def refresh_models(self):
        """重新扫描模型：新放进 models/ 的会出现；重新训练过的 ONNX 会清掉旧预测；交叉验证结果显示为只读模型。"""
        with self.lock:
            found = discover_models()
            for name, (factory, src) in found.items():
                if self.job['running'] and self.job['model'] == name:
                    continue
                stale = False
                if src:
                    marker = self._pred_file(name).with_suffix('.model')
                    stale = self._pred_file(name).exists() and (
                        not marker.exists() or marker.read_text(encoding='utf-8') != _mtime(src))
                if stale:
                    self._pred_file(name).unlink(missing_ok=True)
                    self._pred_file(name).with_suffix('.model').unlink(missing_ok=True)
                    self.models.pop(name, None)
                    self.preds[name] = {}
                    print(f'模型 {name} 已更新，旧的预测已清除，需要重新预测')
                if name not in self.factories or stale:
                    self.factories[name] = factory
                    self.model_src[name] = src
                    self.preds.setdefault(name, self._load_preds(name))
                    self.models.pop(name, None)
            for name in [n for n in self.factories if n not in found and self.job['model'] != n]:
                self.factories.pop(name)
                self.models.pop(name, None)
                self.preds.pop(name, None)
            # 只读模型：predictions/ 里有缓存、但没有对应可运行模型的
            for pf in sorted(self.pred_dir.glob('*.json')):
                name = pf.stem
                if name in self.factories:
                    continue
                m = _mtime(pf)
                if self.readonly.get(name) != m:
                    self.readonly[name] = m
                    self.preds[name] = self._load_preds(name)
            for name in [n for n in self.readonly if not self._pred_file(n).exists()]:
                self.readonly.pop(name)
                self.preds.pop(name, None)

    def model_names(self):
        return list(self.factories) + [n for n in self.readonly if n not in self.factories]

    def files(self):
        return [rel(self.root, p) for f in FOLDERS for p in list_images(self.root / f)]

    def sync(self):
        with self.lock:
            if import_ok(self.root, self.labels):
                save_labels(self.labels_path, self.labels)

    def model(self, name):
        if name not in self.models:
            self.models[name] = self.factories[name]()
        return self.models[name]

    # ---- 接口 ----
    def items(self):
        self.sync()
        out = []
        for f in self.files():
            folder = f.split('/', 1)[0]
            lab = self.labels.get(f)
            m = LEADING_DIGITS.match(Path(f).name) if folder == 'wrong' else None
            out.append({
                'file': f, 'folder': folder,
                'label': lab['label'] if lab else None,
                'source': lab['source'] if lab else None,
                'typed': m.group(1) if m else None,
                'preds': {n: self.preds[n][f] for n in self.preds if f in self.preds[n]},
            })
        return out

    def model_info(self, name):
        """给界面显示的一句话说明。"""
        if name in self.readonly:
            return '只读：交叉验证的预测结果' if name.endswith('_cv') else '只读：只有预测缓存'
        src = self.model_src.get(name)
        if src:
            meta_p = src.with_suffix('.json')
            try:
                m = json.loads(meta_p.read_text(encoding='utf-8'))
                return f'自己训练 · {m.get("trained_at", "")} · {m.get("samples", "?")} 张'
            except (OSError, ValueError):
                return '自定义 ONNX'
        return '现成模型'

    def state(self):
        self.refresh_models()
        files = self.files()
        fs = set(files)
        labs = [r for k, r in self.labels.items() if k in fs]
        return {
            'dir': str(self.root),
            'total': len(files),
            'labeled': sum(1 for r in labs if r['label']),
            'skipped': sum(1 for r in labs if r['source'] == 'skip'),
            'by_source': {s: sum(1 for r in labs if r['source'] == s) for s in ('ok', 'manual', 'auto', 'skip')},
            'models': [{'name': n, 'runnable': n in self.factories, 'info': self.model_info(n),
                        'conf_th': conf_threshold(n, bool(self.model_src.get(n))),
                        'predicted': sum(1 for f in files if f in self.preds.get(n, {}))}
                       for n in self.model_names()],
            'job': dict(self.job),
        }

    def set_label(self, f, label, source):
        with self.lock:
            if source == 'clear':
                self.labels.pop(f, None)
            else:
                self.labels[f] = {'file': f, 'label': label, 'source': source}
            save_labels(self.labels_path, self.labels)

    def evaluate(self):
        """用人工确认过的标注（ok + manual）评估每个模型。"""
        truth = {k: r['label'] for k, r in self.labels.items() if r['source'] in ('ok', 'manual') and r['label']}
        res = []
        for n, preds in self.preds.items():
            rows = [(preds[k][0] == v, preds[k][1]) for k, v in truth.items() if k in preds]
            if not rows:
                res.append({'name': n, 'n': 0})
                continue
            buckets = []
            for th in THRESHOLDS:
                sub = [c for c, conf in rows if conf >= th]
                buckets.append({'th': th, 'coverage': len(sub) / len(rows), 'acc': (sum(sub) / len(sub)) if sub else None})
            res.append({'name': n, 'n': len(rows), 'acc': sum(c for c, _ in rows) / len(rows), 'buckets': buckets})
        return res

    def bulk_accept(self, name, th):
        n = 0
        with self.lock:
            for f in self.files():
                if f in self.labels or f not in self.preds.get(name, {}):
                    continue
                text, conf = self.preds[name][f]
                m = LEADING_DIGITS.match(Path(f).name) if f.startswith('wrong/') else None
                if conf >= th and len(text) == 4 and text.isdigit() and not (m and m.group(1) == text):
                    self.labels[f] = {'file': f, 'label': text, 'source': 'auto'}
                    n += 1
            save_labels(self.labels_path, self.labels)
        return n

    def bulk_accept_agree(self, names):
        """所选的几个模型结果完全一致（且是 4 位数字）的未标注图片，直接采用。"""
        n = 0
        with self.lock:
            for f in self.files():
                if f in self.labels:
                    continue
                texts = {self.preds.get(m, {}).get(f, [None])[0] for m in names}
                if len(texts) != 1:
                    continue
                text = texts.pop()
                m = LEADING_DIGITS.match(Path(f).name) if f.startswith('wrong/') else None
                if text and len(text) == 4 and text.isdigit() and not (m and m.group(1) == text):
                    self.labels[f] = {'file': f, 'label': text, 'source': 'auto'}
                    n += 1
            save_labels(self.labels_path, self.labels)
        return n

    def start_job(self, name, redo=False):
        if self.job['running']:
            return False
        if redo:
            self.preds[name] = {}
        todo = [f for f in self.files() if f not in self.preds[name]]
        self.job = {'running': True, 'model': name, 'done': 0, 'total': len(todo), 'error': ''}

        def work():
            try:
                model = self.model(name)
                for i, f in enumerate(todo, 1):
                    try:
                        self.preds[name][f] = list(model.predict(self.root / f))
                    except Exception as e:  # 单张坏图不影响整体
                        self.preds[name][f] = ['', 0.0]
                        print(f'[{name}] {f}: {e}')
                    self.job['done'] = i
                    if i % 100 == 0:
                        self._save_preds(name)
                self._save_preds(name)
            except Exception as e:
                self.job['error'] = f'{type(e).__name__}: {e}'
                traceback.print_exc()
            finally:
                self.job['running'] = False

        threading.Thread(target=work, daemon=True).start()
        return True


# ---------------- HTTP ----------------

def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 安静一点
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

        def _file(self, path: Path, base: Path):
            try:
                path = path.resolve()
                if not path.is_relative_to(base.resolve()) or not path.is_file():
                    raise FileNotFoundError
                ctype = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                if ctype.startswith('text/') or ctype == 'application/javascript':
                    ctype += '; charset=utf-8'
                self._send(200, path.read_bytes(), ctype)
            except (FileNotFoundError, ValueError):
                self._send(404, b'not found', 'text/plain')

        def do_GET(self):
            u = urlparse(self.path)
            p = unquote(u.path)
            try:
                if p == '/':
                    return self._file(WEB_DIR / 'index.html', WEB_DIR)
                if p.startswith('/img/'):
                    return self._file(store.root / p[5:], store.root)
                if p == '/api/state':
                    return self._json(store.state())
                if p == '/api/items':
                    return self._json(store.items())
                if p == '/api/eval':
                    return self._json(store.evaluate())
                return self._file(WEB_DIR / p.lstrip('/'), WEB_DIR)
            except Exception as e:
                traceback.print_exc()
                self._json({'error': str(e)}, 500)

        def do_POST(self):
            p = urlparse(self.path).path
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                if p == '/api/label':
                    store.set_label(body['file'], body.get('label', ''), body['source'])
                    return self._json({'ok': True})
                if p == '/api/run':
                    if body['model'] not in store.factories:
                        return self._json({'error': '未知模型'}, 400)
                    ok = store.start_job(body['model'], bool(body.get('redo')))
                    return self._json({'ok': ok, 'error': '' if ok else '已有任务在运行'})
                if p == '/api/accept':
                    if body.get('mode') == 'agree':
                        names = [m for m in body.get('models', []) if m in store.preds]
                        if len(names) < 2:
                            return self._json({'error': '至少选两个模型'}, 400)
                        return self._json({'n': store.bulk_accept_agree(names)})
                    return self._json({'n': store.bulk_accept(body['model'], float(body['threshold']))})
                self._json({'error': 'not found'}, 404)
            except Exception as e:
                traceback.print_exc()
                self._json({'error': str(e)}, 500)

    return H


def main():
    ap = argparse.ArgumentParser(description='J-PlatPat 验证码标注（网页版）')
    ap.add_argument('--dir', type=Path, default=None, help='默认自动找 <下载目录>\\jpp_captcha')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    data_dir = args.dir or default_data_dir()
    if not data_dir.is_dir():
        raise SystemExit(f'找不到数据目录：{data_dir}\n先用插件收集一些验证码，或用 --dir 指定位置。')
    store = Store(data_dir)

    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(store))
            break
        except OSError:
            continue
    else:
        raise SystemExit('找不到可用端口')

    url = f'http://127.0.0.1:{port}/'
    print(f'数据目录：{data_dir}')
    print(f'可用模型：{", ".join(store.factories) or "（无）"}')
    print(f'标注页面：{url}    （关闭这个窗口即停止）')
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

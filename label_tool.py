"""J-PlatPat 验证码标注工具

ddddocr 先给每张图预测 4 位数字，你逐张确认：对就回车，错就直接输入正确的数字。

用法:
  python label_tool.py                        # 默认目录：下载目录\\jpp_captcha
  python label_tool.py --dir D:\\xxx\\jpp_captcha
  python label_tool.py --stats                # 只做预测，并用 ok/ 评估 ddddocr 准确率，不开窗口
  python label_tool.py --auto-accept 0.995    # 置信度 ≥ 阈值的预测直接采用（先用 --stats 看准确率再决定）

快捷键:
  Enter        确认当前输入（必须是 4 位数字）
  直接打数字    覆盖预测（预测文字默认全选）
  Esc          跳过：看不清或坏图，不进训练集
  ← / PageUp   回到上一张修改

结果写入 <dir>/labels.csv（file,label,source）。
  source = ok      插件判定验证通过（文件名开头即答案，自动导入）
         = manual  你确认或修改过
         = auto    --auto-accept 自动采用
         = skip    跳过（label 为空）
"""
import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

IMG_EXT = {'.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp'}
LEADING_DIGITS = re.compile(r'^(\d{4})_')
FIELDS = ['file', 'label', 'source']


# ---------- 数据 ----------

def default_data_dir() -> Path:
    """插件存图的位置：<浏览器下载目录>\\jpp_captcha。依次看 Chrome 设置、Windows“下载”文件夹、~/Downloads。"""
    import os
    candidates = []
    local = os.environ.get('LOCALAPPDATA')
    if local:
        for prefs in sorted(Path(local, 'Google', 'Chrome', 'User Data').glob('*/Preferences')):
            try:
                d = json.loads(prefs.read_text(encoding='utf-8')).get('download', {}).get('default_directory')
                if d:
                    candidates.append(Path(d))
            except (OSError, ValueError):
                pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as k:
            v, _ = winreg.QueryValueEx(k, '{374DE290-123F-4565-9164-39C4925E467B}')
            candidates.append(Path(os.path.expandvars(v)))
    except (OSError, ImportError):
        pass
    candidates.append(Path.home() / 'Downloads')
    for c in candidates:
        if (c / 'jpp_captcha').is_dir():
            return c / 'jpp_captcha'
    return candidates[0] / 'jpp_captcha'

def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding='utf-8', newline='') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def save_labels(path: Path, labels: dict):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(labels):
            w.writerow({f: labels[k].get(f, '') for f in FIELDS})
    tmp.replace(path)


def import_ok(root: Path, labels: dict) -> int:
    """ok/ 里的文件名开头就是正确答案，直接导入。"""
    n = 0
    for p in list_images(root / 'ok'):
        m = LEADING_DIGITS.match(p.name)
        key = rel(root, p)
        if m and key not in labels:
            labels[key] = {'file': key, 'label': m.group(1), 'source': 'ok'}
            n += 1
    return n


# ---------- ddddocr 预测 ----------

def load_ocr():
    import ddddocr
    ocr = ddddocr.DdddOcr(show_ad=False)
    ocr.set_ranges('0123456789')  # 1.6.1 的 set_ranges(0) 会返回空串，必须用字符串形式
    return ocr


def predict(ocr, path: Path):
    buf = io.BytesIO()
    Image.open(path).convert('RGB').save(buf, 'PNG')
    r = ocr.classification(buf.getvalue(), probability=True)
    return r['text'], float(r['confidence'])


def run_predictions(root: Path, files, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    todo = [p for p in files if rel(root, p) not in cache]
    if todo:
        print(f'ddddocr 预测 {len(todo)} 张…')
        ocr = load_ocr()
        for i, p in enumerate(todo, 1):
            cache[rel(root, p)] = predict(ocr, p)
            if i % 50 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)}')
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return cache


def report_baseline(root: Path, preds: dict):
    """用 ok/（答案已知）评估 ddddocr，帮你决定 --auto-accept 的阈值。"""
    rows = []
    for p in list_images(root / 'ok'):
        m = LEADING_DIGITS.match(p.name)
        if m and rel(root, p) in preds:
            text, conf = preds[rel(root, p)]
            rows.append((text == m.group(1), conf))
    if not rows:
        print('ok/ 里还没有样本，无法评估 ddddocr 准确率。')
        return
    acc = sum(c for c, _ in rows) / len(rows)
    print(f'\nddddocr 在 ok/ 上的整串准确率：{acc:.1%}（{len(rows)} 张）')
    for th in (0.9, 0.95, 0.98, 0.99, 0.995):
        sub = [c for c, conf in rows if conf >= th]
        if sub:
            print(f'  置信度 ≥ {th}: 覆盖 {len(sub) / len(rows):5.1%}，其中准确率 {sum(sub) / len(sub):.1%}')


# ---------- 界面 ----------

class App:
    SCALE = 3

    def __init__(self, root: Path, items, labels: dict, preds: dict, labels_path: Path):
        import tkinter as tk
        self.tk = tk
        self.root, self.items, self.labels, self.preds, self.labels_path = root, items, labels, preds, labels_path
        self.idx = 0
        self._confirm = None  # (key, value)：wrong/ 里输入了当时被判错的值，需要二次确认

        self.win = tk.Tk()
        self.win.title('J-PlatPat 验证码标注')
        self.win.configure(padx=16, pady=12)
        self.progress = tk.Label(self.win, font=('Microsoft YaHei', 11))
        self.progress.pack(anchor='w')
        self.canvas = tk.Label(self.win, bd=1, relief='solid')
        self.canvas.pack(pady=8)
        self.info = tk.Label(self.win, font=('Microsoft YaHei', 10), justify='left')
        self.info.pack(anchor='w')
        self.entry = tk.Entry(self.win, font=('Consolas', 36), width=6, justify='center')
        self.entry.pack(pady=8)
        self.msg = tk.Label(self.win, font=('Microsoft YaHei', 10), fg='#c00')
        self.msg.pack()
        tk.Label(self.win, fg='#666', font=('Microsoft YaHei', 9),
                 text='Enter 确认   直接打字覆盖   Esc 跳过（看不清）   ← / PageUp 上一张').pack(pady=(6, 0))

        self.entry.bind('<Return>', lambda e: self.accept())
        self.entry.bind('<KP_Enter>', lambda e: self.accept())
        self.win.bind('<Escape>', lambda e: self.skip())
        self.win.bind('<Prior>', lambda e: self.back())
        self.entry.bind('<Left>', lambda e: self.back() if self.entry.index('insert') == 0 else None)
        self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.show()

    def show(self):
        tk = self.tk
        if self.idx >= len(self.items):
            self.progress.config(text=f'全部完成：{len(self.items)} 张')
            self.canvas.config(image='', text='没有待标注的图片了，可以关闭窗口', width=60, height=6)
            self.entry.delete(0, 'end')
            self.info.config(text='')
            return
        key = self.items[self.idx]
        p = self.root / key
        im = Image.open(p).convert('RGB')
        im = im.resize((im.width * self.SCALE, im.height * self.SCALE), Image.LANCZOS)
        from PIL import ImageTk
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.config(image=self._photo, text='', width=im.width, height=im.height)

        done = sum(1 for k in self.items if k in self.labels)
        self.progress.config(text=f'第 {self.idx + 1} / {len(self.items)} 张    已处理 {done}')

        pred, conf = self.preds.get(key, ('', 0.0))
        lines = [f'{key}', f'ddddocr 预测：{pred or "（空）"}    置信度 {conf:.3f}']
        color = '#080' if conf >= 0.99 else '#c70' if conf >= 0.9 else '#c00'
        wrong_typed = LEADING_DIGITS.match(Path(key).name) if key.startswith('wrong/') else None
        if wrong_typed:
            lines.append(f'⚠ 当时输入了 {wrong_typed.group(1)}，那是错误答案')
            if pred == wrong_typed.group(1):
                color = '#c00'
        self.info.config(text='\n'.join(lines), fg=color)

        existing = self.labels.get(key)
        value = existing['label'] if existing and existing['label'] else (pred if len(pred) == 4 else '')
        self.entry.delete(0, 'end')
        self.entry.insert(0, value)
        self.entry.select_range(0, 'end')
        self.entry.focus_set()
        self.msg.config(text='')

    def _set(self, label, source):
        key = self.items[self.idx]
        self.labels[key] = {'file': key, 'label': label, 'source': source}
        save_labels(self.labels_path, self.labels)
        self.idx += 1
        self.show()

    def accept(self):
        if self.idx >= len(self.items):
            return
        v = self.entry.get().strip()
        if not re.fullmatch(r'\d{4}', v):
            self.msg.config(text='需要正好 4 位数字')
            return
        key = self.items[self.idx]
        m = LEADING_DIGITS.match(Path(key).name) if key.startswith('wrong/') else None
        if m and v == m.group(1) and self._confirm != (key, v):
            self._confirm = (key, v)
            self.msg.config(text='这和当时被判错的输入相同，确定吗？再按一次 Enter 确认')
            return
        self._confirm = None
        self._set(v, 'manual')

    def skip(self):
        if self.idx < len(self.items):
            self._set('', 'skip')

    def back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show()
        return 'break'

    def close(self):
        save_labels(self.labels_path, self.labels)
        self.win.destroy()

    def run(self):
        self.win.mainloop()


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser(description='J-PlatPat 验证码标注工具')
    ap.add_argument('--dir', type=Path, default=None, help='默认自动找 <下载目录>\\jpp_captcha')
    ap.add_argument('--stats', action='store_true', help='只预测并评估，不开窗口')
    ap.add_argument('--auto-accept', type=float, metavar='CONF', help='置信度 ≥ CONF 的预测直接采用')
    args = ap.parse_args()

    root: Path = args.dir or default_data_dir()
    if not root.is_dir():
        sys.exit(f'找不到目录：{root}\n先用插件收集一些验证码，或用 --dir 指定位置。')
    labels_path = root / 'labels.csv'
    labels = load_labels(labels_path)

    n_ok = import_ok(root, labels)
    if n_ok:
        print(f'从 ok/ 导入 {n_ok} 张已验证样本')

    pending = list_images(root / 'unlabeled') + list_images(root / 'wrong')
    preds = run_predictions(root, pending + list_images(root / 'ok'), root / 'predictions.json')
    report_baseline(root, preds)

    items = [rel(root, p) for p in pending if rel(root, p) not in labels]

    if args.auto_accept is not None:
        n = 0
        for key in items:
            text, conf = preds[key]
            m = LEADING_DIGITS.match(Path(key).name) if key.startswith('wrong/') else None
            if conf >= args.auto_accept and len(text) == 4 and not (m and m.group(1) == text):
                labels[key] = {'file': key, 'label': text, 'source': 'auto'}
                n += 1
        print(f'\n--auto-accept {args.auto_accept}：自动采用 {n} 张')
        items = [k for k in items if k not in labels]

    save_labels(labels_path, labels)
    total = sum(1 for r in labels.values() if r['label'])
    print(f'\n已有可用标注 {total} 张，待人工确认 {len(items)} 张')

    if args.stats or not items:
        return
    App(root, items, labels, preds, labels_path).run()
    total = sum(1 for r in labels.values() if r['label'])
    print(f'已保存。可用标注共 {total} 张 → {labels_path}')


if __name__ == '__main__':
    main()

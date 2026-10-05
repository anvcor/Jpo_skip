"""把训练好的 PyTorch 模型导出成插件能直接用的格式（纯 JavaScript 推理，不需要 onnxruntime）。

  python export_web.py            # 读 models/jpp_cnn.pt → 写 extension/model/jpp_cnn.bin + jpp_cnn.json

train.py 训练完会自动调用这里，一般不用手动运行。

格式：
  jpp_cnn.bin   所有权重按顺序拼接的 float32（小端）。卷积层已经把 BatchNorm 折进去了（w·γ/σ，b = β − μ·γ/σ）
  jpp_cnn.json  每一层的类型、形状、在 .bin 里的偏移，以及预处理参数（尺寸、裁边、归一化）
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'extension' / 'model'


def fold_bn(conv, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w.detach().cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)


def export(net, extra_meta=None, out_dir: Path = OUT_DIR):
    import train as T
    net = net.cpu().eval()
    layers, blobs, offset = [], [], 0

    def add(arr):
        nonlocal offset
        arr = np.ascontiguousarray(arr, dtype='<f4')
        blobs.append(arr.tobytes())
        start = offset
        offset += arr.size
        return start

    for blk in net.features:                      # 每个 block：conv-bn-relu-conv-bn-relu-maxpool
        mods = list(blk)
        for conv, bn in ((mods[0], mods[1]), (mods[3], mods[4])):
            w, b = fold_bn(conv, bn)
            layers.append({'type': 'conv3x3_relu', 'in': int(w.shape[1]), 'out': int(w.shape[0]),
                           'w': add(w), 'b': add(b)})
        layers.append({'type': 'maxpool2'})
    layers.append({'type': 'avgpool_rows'})       # AdaptiveAvgPool2d((1, 15))：高 3 → 1，宽本来就是 15
    fc1, fc2 = net.head[2], net.head[5]
    layers.append({'type': 'fc_relu', 'in': fc1.in_features, 'out': fc1.out_features,
                   'w': add(fc1.weight.detach().numpy()), 'b': add(fc1.bias.detach().numpy())})
    layers.append({'type': 'fc', 'in': fc2.in_features, 'out': fc2.out_features,
                   'w': add(fc2.weight.detach().numpy()), 'b': add(fc2.bias.detach().numpy())})

    meta = {
        'name': T.NAME, 'height': T.H, 'width': T.W, 'crop': T.CROP, 'mean': T.MEAN, 'std': T.STD,
        'length': T.LENGTH, 'classes': T.CLASSES, 'charset': '0123456789',
        'resize': 'pil_bilinear', 'grayscale': 'pil_L',
        'floats': offset, 'layers': layers, 'exported_at': time.strftime('%Y-%m-%d %H:%M'),
        **(extra_meta or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{T.NAME}.bin').write_bytes(b''.join(blobs))
    (out_dir / f'{T.NAME}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return out_dir / f'{T.NAME}.bin', offset


def main():
    import train as T
    net = T.CaptchaNet()
    net.load_state_dict(torch.load(T.MODELS_DIR / f'{T.NAME}.pt', map_location='cpu'))
    onnx_meta = T.MODELS_DIR / f'{T.NAME}.json'
    extra = {}
    if onnx_meta.exists():
        m = json.loads(onnx_meta.read_text(encoding='utf-8'))
        extra = {k: m[k] for k in ('trained_at', 'samples') if k in m}
    path, n = export(net, extra)
    print(f'已导出 {n:,} 个参数 → {path}（{path.stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()

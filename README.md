# J-PlatPat 验证码助手

在 [J-PlatPat](https://www.j-platpat.inpit.go.jp/)（日本特许厅专利检索平台）下载公报时，会弹出 4 位数字的图片验证码，PDF 还会在单独的查看器窗口里打开。
这个项目包含一个 Chrome 插件：**你点「文献単位PDF」之后，验证码自动识别并提交，PDF 直接存到下载目录**，不再弹窗口。另外还包含从收集数据、标注、训练到评估的完整工具链。

- 自己训练的小 CNN（0.96M 参数），5 折交叉验证 **3,031 张 0 错**；在人工确认的难例上 ddddocr 是 93.6%
- 纯 JavaScript 推理，不依赖 onnxruntime，每张约 0.2 秒，**不联网、不上传任何数据**
- 训练全过程和评估方法见 **[docs/TRAINING.md](docs/TRAINING.md)**

## 目录

```
extension/            Chrome 插件（Manifest V3）
  model.js            纯 JS 推理（逐像素复刻 PIL 的预处理）
  model/              导出的模型权重（train.py 自动更新）
  content.js          识别 / 填入 / 自动提交 / 自动点 PDF 下载 / 收集
  page_hook.js        拦截 PDF 查看器窗口，改成直接下载
  background.js       下载 PDF、保存图片、统计实测准确率
label_server.py       本地标注网页的后端（只用标准库）
label_web/            标注网页前端
label_tool.py         命令行版标注 + 公共函数
train.py              训练 / 交叉验证 / 导出
export_web.py         PyTorch → 插件权重
models/               训练好的模型（ONNX、PyTorch）和报告
docs/TRAINING.md      训练记录
*.bat                 Windows 双击启动：标注、训练、交叉验证
```

## 快速开始

### 只用插件

1. Chrome 打开 `chrome://extensions`，开启「开发者模式」
2. 「加载已解压的扩展程序」→ 选择 `extension/` 文件夹
3. 刷新 J-PlatPat 页面，照常点「文献単位PDF」，文件会出现在 `下载目录\J-PlatPat\`

### 自己训练

需要 Python 3.11、PyTorch 2.x（有 CUDA 更快）、`numpy==1.26.4`、Pillow、`onnx<1.19`、onnxruntime；可选 ddddocr、`ppllocr`（用 `--no-deps` 安装）。

```bash
python label_server.py      # 标注（或双击 start_label.bat）
python train.py             # 训练，同时更新 models/ 和 extension/model/（或双击 train.bat）
python train.py --cv 5      # 交叉验证（或双击 cross_validate.bat）
```

数据默认放在 `<浏览器下载目录>/jpp_captcha/`（插件收集的位置），可以用 `--dir` 指定。

## 使用须知

- 自动提交只在**你亲手点了下载**之后才会发生，插件自己从不点下载按钮。请遵守 J-PlatPat 的使用条款，不要用来做高频、批量的自动访问。
- 仓库不包含验证码图片数据集。
- 与日本特许厅、INPIT 没有任何关系。

## License

[MIT](LICENSE)

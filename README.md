# 详情图层分离 ComfyUI 节点 (ComfyUI-LayerSeparation)

> 一个把电商详情图「一键拆成可编辑图层」的 ComfyUI 自定义节点：背景、前景元素、文字三层自动分离，全中文界面，开箱即用。

把一张电商详情图分离成三层:

- **背景层** — LaMa inpaint 抹掉前景/文字后的干净底图
- **前景元素** — BiRefNet/RMBG/u2net 抠出的商品/元素 (IMAGE batch)
- **前景蒙版** — 对应每个元素的 alpha (MASK batch)
- **manifest** — 画布尺寸 + 每个元素的 bbox + 文字信息 (JSON 字符串)

节点是自包含的标准 ComfyUI V1 插件, 重模型 (OCR / rembg / LaMa) 用模块级懒加载单例, 跨次执行复用不重载。

## 安装

进入 ComfyUI 的 `custom_nodes/` 目录, clone 本仓库并装依赖:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zriyox/ComfyUI-LayerSeparation.git
cd ComfyUI-LayerSeparation
pip install -r requirements.txt
```

装完重启 ComfyUI, 节点出现在 `image/layer_separation` 分类下 (中文「详情图层分离」「保存文本」)。

### GPU 加速 (推荐)

`requirements.txt` 默认装的是 CPU 版 onnxruntime。GPU 环境跑 rembg/OCR 想用上显卡:

```bash
pip uninstall onnxruntime -y && pip install onnxruntime-gpu
```

pipeline 会按可用 provider 自动选 `CUDA > CoreML > CPU`, 无 GPU 时自动回退, 不会报错。

> ⚠️ 若 `pip install -r requirements.txt` 把 ComfyUI 的 torch 顶成了别的版本 (simple-lama-inpainting 可能拉 torch 依赖), 改用 `pip install --no-deps simple-lama-inpainting` 单独装它, 其余依赖正常装。

## 首次运行会自动下载模型

| 模型 | 大小 | 来源 | 缓存位置 |
|------|------|------|----------|
| big-lama.pt (LaMa) | ~196M | GitHub release | `~/.cache/torch/hub/checkpoints/` |
| birefnet / RMBG / u2net / isnet | ~176M–972M+ | rembg | `$U2NET_HOME` (默认 `~/.u2net/`) |
| RapidOCR | 随包自带 | pip | 包内 |

首跑需联网拉模型, 之后复用。离线环境请提前把模型放到上述目录。

**预热下载(推荐)**: 部署后先跑一次, 避免首次执行节点时卡几分钟下模型:

```bash
cd ComfyUI/custom_nodes/ComfyUI-LayerSeparation
python download_models.py                 # 预热默认 birefnet-general + LaMa
python download_models.py bria-rmbg       # 或指定其它抠图模型
```

## VLM 文字分类 (可选)

`use_vlm` 开启时, 用阿里云 DashScope 的 qwen-vl 对文字做分类, 需要密钥:

- 节点上 `dashscope_api_key` 直接填, 或
- 设环境变量 `DASHSCOPE_API_KEY`, 或
- 在本插件目录放 `.env` 写 `DASHSCOPE_API_KEY=xxx`

关闭 `use_vlm` 则纯 OCR, 不联网、不需要密钥。

## 中文显示

节点标题、所有输入/输出端口名、参数提示均为中文, 翻译在 `locales/zh/`, ComfyUI 服务端自动加载 (前端语言切到中文即可)。

## 参数

| 输入 | 类型 | 默认 | 说明 |
|------|------|------|------|
| 图像 | IMAGE | — | 输入的详情图。支持 batch: 多张会逐帧分离, 不再只取第 0 帧 |
| VLM文字分类 | BOOLEAN | 开 | 开=qwen-vl 分类(联网), 关=纯 OCR |
| 抠图模型 | 选择 | birefnet-general | birefnet-general / birefnet-massive / bria-rmbg / isnet-general-use / u2net / u2netp |
| DashScope密钥 | STRING | 空 | qwen-vl 密钥, 留空走 env/.env |
| min_area | FLOAT | 0.0015 | 最小元素面积占比。**详情图拆不出多元素时调小**(保留更多小元素) |
| close_ksize | INT | 3 | 形态学闭运算核。**0/1=关闭**(避免相邻元素被粘成一团); 越大越易合并 |
| alpha_thr | INT | 30 | 前景 alpha 二值化阈值, 越高越严格 |
| ocr_min_score | FLOAT | 0.5 | OCR 置信度下限, 低于此丢弃 |
| vlm_model | STRING | qwen-vl-max | VLM 模型名(DashScope), 可改 qwen-vl-plus 等。也可用 env `DASHSCOPE_VLM_MODEL` / `DASHSCOPE_VLM_ENDPOINT` |
| element_mode | 选择 | canvas | 前景元素输出模式。canvas=元素贴回原画布(便于合成, 4K 内存大) / cropped=输出裁剪小图(省内存, 原位见 bbox) |
| mask_mode | 选择 | canvas | 前景蒙版输出模式。canvas=蒙版贴回原画布(便于直接接遮罩节点) / cropped=输出裁剪范围内的蒙版(省内存) |

> **多图说明**: `前景元素` 输出本就是 batch, 一个独立元素一张图。若拆出的元素偏少, 调小 `min_area`、把 `close_ksize` 设 0~1 即可拆得更细。

| 输出 | 类型 | 说明 |
|------|------|------|
| 背景层 | IMAGE | 干净背景 (batch, 与输入帧数一致) |
| 前景元素 | IMAGE | 抠出的元素 (batch, 跨帧拼接) |
| 前景蒙版 | MASK | 元素 alpha (batch) |
| manifest | STRING | 画布/bbox/文字的 JSON。含 `meta.vlm_status`(ok/skipped/failed/disabled); 单图为对象, 多图为数组 |

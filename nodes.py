#!/usr/bin/env python3
"""详情图层分离 ComfyUI 节点。

把仓库根目录的 pipeline.run_pipeline 封装成单个 ComfyUI 节点:
输入一张 IMAGE, 输出 背景层(IMAGE) / 前景元素(IMAGE batch) / 前景蒙版(MASK batch) / manifest(JSON STRING)。
重模型(OCR/rembg/LaMa)沿用 pipeline 的模块级懒加载单例, 跨次执行复用不重载。
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# pipeline.py 已 vendor 进本插件目录, 包内相对导入, 插件自包含。
from . import pipeline

# rembg 支持的抠图模型(前景层)。birefnet 系列质量最好, u2net/isnet 更快更省显存。
_FG_MODELS = ["birefnet-general", "birefnet-massive", "isnet-general-use", "u2net", "u2netp"]


class LayerSeparationNode:
    """详情图层分离: 背景(LaMa inpaint) + 前景元素(BiRefNet 抠图) + 文字(OCR/可选VLM)。"""

    CATEGORY = "image/layer_separation"
    FUNCTION = "separate"
    # 末尾追加两张全画布并集遮罩(fg_mask / fg_text_mask), 不动前 4 个口的槽位, 现有连线不断。
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "MASK", "MASK")
    RETURN_NAMES = ("background", "elements", "element_masks", "manifest", "fg_mask", "fg_text_mask")
    # elements/element_masks 是逐元素 list(每张各自真实尺寸); 其余(含两张全画布遮罩)是单值/batch。
    OUTPUT_IS_LIST = (False, True, True, False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入的详情图(IMAGE)。"}),
            },
            "optional": {
                "use_vlm": ("BOOLEAN", {"default": True, "label_on": "VLM文字分类开", "label_off": "VLM关"}),
                "fg_model": (_FG_MODELS, {
                    "default": "birefnet-general",
                    "tooltip": "前景抠图模型。birefnet 系列质量最好, u2net/isnet 更快更省显存。",
                }),
                "dashscope_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "DashScope(qwen-vl) 密钥。留空则回退到环境变量 DASHSCOPE_API_KEY 或仓库 .env。use_vlm 关时此项忽略。",
                }),
                "min_area": ("FLOAT", {
                    "default": 0.0015, "min": 0.0, "max": 0.2, "step": 0.0005, "round": 0.0001,
                    "tooltip": "最小元素面积占比。越小保留越多小元素(也更易出噪点)。详情图拆不出多元素时调小。",
                }),
                "close_ksize": ("INT", {
                    "default": 3, "min": 0, "max": 25, "step": 1,
                    "tooltip": "形态学闭运算核大小。0/1=关闭(避免把相邻元素粘成一团); 越大越易把多个元素合并成一个。",
                }),
                "alpha_thr": ("INT", {
                    "default": 30, "min": 0, "max": 254, "step": 1,
                    "tooltip": "前景 alpha 二值化阈值。越高越严格(只留实心主体)。",
                }),
                "ocr_min_score": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "OCR 置信度下限, 低于此的文字丢弃。",
                }),
                "vlm_model": ("STRING", {
                    "default": "qwen-vl-max", "multiline": False,
                    "tooltip": "VLM 模型名(DashScope), 如 qwen-vl-max / qwen-vl-plus。use_vlm 关时忽略。",
                }),
                "element_mode": (["canvas", "cropped"], {
                    "default": "canvas",
                    "tooltip": "canvas=元素贴回原画布(便于直接合成, 4K 多元素内存大); cropped=只输出裁剪小图(省内存, 原位置见 manifest bbox)。",
                }),
            },
        }

    # ---------------- tensor <-> 文件/PIL ----------------
    @staticmethod
    def _frame_to_pil(frame):
        """单帧 IMAGE [H,W,3] float0-1 -> PIL RGB。"""
        arr = (frame.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def _pil_to_image_tensor(pil_rgb):
        """PIL RGB -> IMAGE tensor [1,H,W,3] float0-1。"""
        arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    @staticmethod
    def _concat_image_batches(batches):
        """把多帧产生的 [n_i,H_i,W_i(,C)] 批次在 batch 维拼成一个; 尺寸不一时右下补零对齐到最大 H/W。
        支持 IMAGE(4D) 与 MASK(3D)。单批次直接返回, 不做无谓拷贝。"""
        batches = [b for b in batches if b is not None and b.shape[0] > 0]
        if not batches:
            return None
        if len(batches) == 1:
            return batches[0]
        maxH = max(t.shape[1] for t in batches)
        maxW = max(t.shape[2] for t in batches)
        out = []
        for t in batches:
            n, h, w = t.shape[0], t.shape[1], t.shape[2]
            if h != maxH or w != maxW:
                shape = (n, maxH, maxW, t.shape[3]) if t.dim() == 4 else (n, maxH, maxW)
                pad = torch.zeros(shape, dtype=t.dtype)
                pad[:, :h, :w, ...] = t
                t = pad
            out.append(t)
        return torch.cat(out, 0)

    def _composite_elements(self, manifest, workdir, element_mode="canvas"):
        """把 N 个 RGBA 前景小图切成 (IMAGE list, MASK list)。每个元素是独立一帧:
          canvas : 按 bbox 贴回 (W,H) 全画布, 与 manifest 渲染语义一致, 便于直接合成。
          cropped: 各自输出自己 bbox 的真实尺寸 (w,h), 不补零到全体最大尺寸(那会把小元素
                   全撑到画布尺寸, 既不省内存也让裁剪图尺寸对不上 bbox)。
        返回 list[ [1,h,w,3] ] / list[ [1,h,w] ], 配合节点 OUTPUT_IS_LIST,
        下游 Preview/Save 逐帧各自原尺寸接收。RGB 在 alpha=0 处清零, 只显示抠图本体。"""
        W = int(manifest["canvas"]["width"])
        H = int(manifest["canvas"]["height"])
        items = [it for it in manifest.get("images", []) if it["bbox"][2] > 0 and it["bbox"][3] > 0]
        if not items:  # N=0 兜底, 给下游一帧全0, 避免空 list 崩
            return [torch.zeros((1, H, W, 3), dtype=torch.float32)], [torch.zeros((1, H, W), dtype=torch.float32)]
        imgs, masks = [], []
        for item in items:
            x, y, w, h = item["bbox"]
            el = Image.open(Path(workdir) / item["url"]).convert("RGBA")
            if el.size != (w, h):
                el = el.resize((w, h), Image.LANCZOS)
            if element_mode == "cropped":
                canvas = el                                      # 元素本体, 真实 (w,h)
            else:
                canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                canvas.paste(el, (int(x), int(y)))
            rgba = np.asarray(canvas, dtype=np.float32) / 255.0  # [h,w,4]
            alpha = rgba[..., 3]
            rgb = rgba[..., :3] * alpha[..., None]               # 透明处清零
            imgs.append(torch.from_numpy(rgb)[None, ...])        # [1,h,w,3]
            masks.append(torch.from_numpy(alpha)[None, ...])     # [1,h,w]
        return imgs, masks

    def _build_full_masks(self, manifest, workdir):
        """构建两张与原图同尺寸 (H,W) 的并集遮罩, 供「遮罩重绘」一次跑完直接用:
          fg_mask      : 仅前景元素 —— 各元素真实 alpha 形状贴回各自 bbox 后取并集(max)。
          fg_text_mask : 前景元素 ∪ OCR 文字 bbox 矩形(文字层无 alpha, 按矩形覆盖)。
        与 manifest/background 同尺寸, 数值 0~1。返回 ([1,H,W], [1,H,W])。"""
        W = int(manifest["canvas"]["width"])
        H = int(manifest["canvas"]["height"])
        elem = np.zeros((H, W), dtype=np.float32)
        for item in manifest.get("images", []):
            x, y, w, h = item["bbox"]
            if w <= 0 or h <= 0:
                continue
            el = Image.open(Path(workdir) / item["url"]).convert("RGBA")
            if el.size != (w, h):
                el = el.resize((w, h), Image.LANCZOS)
            a = np.asarray(el, dtype=np.float32)[..., 3] / 255.0  # [h,w] alpha
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
            if x1 <= x0 or y1 <= y0:
                continue
            region = elem[y0:y1, x0:x1]
            np.maximum(region, a[: y1 - y0, : x1 - x0], out=region)  # 并集
        elem_text = elem.copy()
        for t in manifest.get("texts", []):
            x, y, w, h = t["bbox"]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
            if x1 > x0 and y1 > y0:
                elem_text[y0:y1, x0:x1] = 1.0                       # 文字按矩形覆盖
        return torch.from_numpy(elem)[None, ...], torch.from_numpy(elem_text)[None, ...]

    # ---------------- 主入口 ----------------
    def separate(self, image, use_vlm=True, fg_model="birefnet-general", dashscope_api_key="",
                 min_area=0.0015, close_ksize=3, alpha_thr=30, ocr_min_score=0.5,
                 vlm_model="qwen-vl-max", element_mode="canvas"):
        # 统一密钥/模型入口: 节点上填了就写进环境变量, pipeline 优先读 env;
        # 留空则沿用 env / 仓库 .env 的现有兜底。
        key = (dashscope_api_key or "").strip()
        if use_vlm and key:
            os.environ["DASHSCOPE_API_KEY"] = key
        if use_vlm and (vlm_model or "").strip():
            os.environ["DASHSCOPE_VLM_MODEL"] = vlm_model.strip()

        fg_kwargs = {"min_area": float(min_area), "close_ksize": int(close_ksize), "alpha_thr": int(alpha_thr)}

        # 遍历整个输入 batch, 逐帧分离; B=1 时与单图行为一致。
        # 元素/蒙版按帧收成扁平 list(每张各自真实尺寸, 不跨帧补零拼接)。
        # fg_mask/fg_text_mask 是全画布单值, 按帧 batch 拼接(与 background 对齐)。
        bgs, elem_list, mask_list, manifests = [], [], [], []
        fgm_list, fgtm_list = [], []
        for b in range(image.shape[0]):
            workdir = tempfile.mkdtemp(prefix="comfy_layersep_")
            input_png = Path(workdir) / "input.png"
            self._frame_to_pil(image[b]).save(input_png)

            manifest = pipeline.run_pipeline(
                str(input_png), workdir, stem="input", use_vlm=use_vlm, fg_model=fg_model,
                fg_kwargs=fg_kwargs, text_min_score=float(ocr_min_score),
            )

            W = int(manifest["canvas"]["width"])
            H = int(manifest["canvas"]["height"])
            bg_pil = Image.open(Path(workdir) / manifest["background"]).convert("RGB")
            if bg_pil.size != (W, H):  # 防御: 背景与 canvas 必须同尺寸
                bg_pil = bg_pil.resize((W, H), Image.LANCZOS)
            bgs.append(self._pil_to_image_tensor(bg_pil))  # [1,H,W,3]
            el, mk = self._composite_elements(manifest, workdir, element_mode=element_mode)
            elem_list.extend(el)
            mask_list.extend(mk)
            fgm, fgtm = self._build_full_masks(manifest, workdir)
            fgm_list.append(fgm)
            fgtm_list.append(fgtm)
            manifests.append(manifest)

        background = self._concat_image_batches(bgs)
        # elements/element_masks 是 list(OUTPUT_IS_LIST), 逐元素各自真实尺寸, 不跨帧补零。
        elements = elem_list
        element_masks = mask_list
        # 全画布并集遮罩: 多帧按 batch 拼接, 与 background 帧对齐。
        fg_mask = self._concat_image_batches(fgm_list)
        fg_text_mask = self._concat_image_batches(fgtm_list)
        # 单图输出 manifest 对象(向后兼容); 多图输出 manifest 数组。
        manifest_json = json.dumps(manifests[0] if len(manifests) == 1 else manifests, ensure_ascii=False)
        return (background, elements, element_masks, manifest_json, fg_mask, fg_text_mask)


class SaveTextNode:
    """把 STRING(如 manifest) 落成 ComfyUI output 目录下的文本文件。

    ComfyUI 的 STRING 连线数据不会被 API 直接回传, 必须由 OUTPUT_NODE 写成 output
    目录里的文件才能被回传/下载。把本节点接在「详情图层分离」的 manifest 输出后面,
    即可拿到 .txt/.json 文件。
    """

    CATEGORY = "image/layer_separation"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "要保存的文本(把 manifest 连进来)。"}),
                "filename_prefix": ("STRING", {"default": "manifest", "tooltip": "文件名前缀, 自动追加递增序号。可含子目录, 如 layersep/manifest。"}),
                "extension": (["txt", "json"], {"default": "txt", "tooltip": "文件扩展名。"}),
            },
        }

    def save(self, text, filename_prefix="manifest", extension="txt"):
        # 延迟导入: ComfyUI 运行时才有 folder_paths, 包外导入本模块不受影响。
        import folder_paths

        out_dir = folder_paths.get_output_directory()
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        filename = f"{fname}_{counter:05d}.{extension}"
        path = os.path.join(full_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
        # ui.text 给前端预览; 文件已落 output, 可被 ComfyUI API 回传/下载。
        return {"ui": {"text": [text], "string": [text],
                       "files": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "LayerSeparation": LayerSeparationNode,
    "LayerSeparationSaveText": SaveTextNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerSeparation": "详情图层分离 (Layer Separation)",
    "LayerSeparationSaveText": "保存文本 (Save Text)",
}

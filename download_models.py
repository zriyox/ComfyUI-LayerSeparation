#!/usr/bin/env python3
"""模型预热脚本: 提前把 rembg(抠图) + LaMa(背景 inpaint) 模型下到本地缓存,
避免节点首次执行时阻塞几分钟下载、让人以为卡死。OCR(RapidOCR)随 pip 包自带, 无需下载。

用法:
  python download_models.py                      # 下默认 birefnet-general + LaMa
  python download_models.py bria-rmbg u2net      # 指定要预热的抠图模型
  RembG 模型可选: birefnet-general / birefnet-massive / bria-rmbg / isnet-general-use / u2net / u2netp
"""
import sys
import os

os.environ.setdefault("U2NET_HOME", os.path.join(os.path.expanduser("~"), ".u2net"))


def pull_rembg(models):
    from rembg import new_session
    for m in models:
        print(f"[rembg] 下载/校验 {m} ...", flush=True)
        new_session(m)            # 不存在则下载到 $U2NET_HOME
        print(f"[rembg] {m} OK", flush=True)


def pull_lama():
    print("[lama] 下载/校验 big-lama.pt ...", flush=True)
    try:
        from simple_lama_inpainting.utils.util import download_model
    except ImportError:
        from simple_lama_inpainting.utils import download_model
    mp = download_model(
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    )
    print(f"[lama] OK -> {mp}", flush=True)


def main():
    models = sys.argv[1:] or ["birefnet-general"]
    pull_rembg(models)
    pull_lama()
    print("全部模型就绪。", flush=True)


if __name__ == "__main__":
    main()

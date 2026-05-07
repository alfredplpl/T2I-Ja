#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import PixArtTransformer2DModel

from t2i_ja import build_pixart_sigma_pipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    config = load_config(args.config)
    transformer = PixArtTransformer2DModel.from_pretrained(args.checkpoint, torch_dtype=dtype)
    pipeline = build_pixart_sigma_pipeline(config, transformer=transformer, dtype=dtype, device=device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

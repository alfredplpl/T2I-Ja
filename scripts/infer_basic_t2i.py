#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import PixArtTransformer2DModel

from t2i_ja import build_pixart_sigma_pipeline, encode_qwen_prompt, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
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
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = encode_qwen_prompt(
        config,
        args.prompt,
        dtype=dtype,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipeline(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        negative_prompt_embeds=negative_embeds,
        negative_prompt_attention_mask=negative_mask,
        height=int(config.image["resolution"]),
        width=int(config.image["resolution"]),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        use_resolution_binning=False,
        generator=generator,
    ).images[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="microsoft/Florence-2-large")
    parser.add_argument("--task", default="<DETAILED_CAPTION>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--relative-to", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def iter_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def jsonl_image_path(path: Path, path_mode: str, relative_to: Path | None) -> str:
    if path_mode == "absolute":
        return str(path.resolve())
    base = relative_to or Path.cwd()
    return str(path.resolve().relative_to(base.resolve()))


def caption_image(
    image_path: Path,
    task: str,
    processor,
    model,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = move_inputs(inputs, device=device, dtype=dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    caption = parsed.get(task, generated_text)
    if isinstance(caption, dict):
        caption = json.dumps(caption, ensure_ascii=False)
    return str(caption).strip()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output = Path(args.output)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    images = iter_images(image_dir, args.recursive)
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_to = Path(args.relative_to) if args.relative_to else None

    with output.open("w", encoding="utf-8") as f:
        for index, image_path in enumerate(images, start=1):
            try:
                text = caption_image(
                    image_path=image_path,
                    task=args.task,
                    processor=processor,
                    model=model,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                record = {
                    "image": jsonl_image_path(image_path, args.path_mode, relative_to),
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{index}/{len(images)}] {image_path} -> {text}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[{index}/{len(images)}] failed: {image_path}: {exc}")


if __name__ == "__main__":
    main()

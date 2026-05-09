#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ORIGINAL_FIND_SPEC = importlib.util.find_spec


def block_flash_attn_for_captioning() -> None:
    if getattr(importlib.util, "_t2i_ja_flash_attn_blocked", False):
        return

    def find_spec_without_flash_attn(name, package=None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        if name == "flash_attn_2_cuda":
            return None
        return ORIGINAL_FIND_SPEC(name, package)

    sys.modules.pop("flash_attn", None)
    sys.modules.pop("flash_attn_2_cuda", None)
    importlib.util.find_spec = find_spec_without_flash_attn
    importlib.util._t2i_ja_flash_attn_blocked = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="MIL-UT/Asagi-2B")
    parser.add_argument(
        "--prompt",
        default="簡潔に説明してください。",
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--relative-to", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]


def model_load_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "device_map": args.device_map,
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


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


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_asagi_prompt(prompt: str) -> str:
    return (
        "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
        f"### 指示:\n<image>\n{prompt}\n\n"
        "### 応答:\n"
    )


def load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def preprocess_asagi_inputs(image_path: Path, prompt: str, processor):
    image = load_rgb_image(image_path)
    text_prompt = build_asagi_prompt(prompt)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    tokenized = processor.tokenizer(text_prompt, return_tensors="pt")
    inputs["input_ids"] = tokenized["input_ids"]
    inputs["attention_mask"] = tokenized["attention_mask"]
    return text_prompt, inputs


def iter_preprocessed_asagi_inputs(
    image_paths: list[Path],
    prompt: str,
    processor,
    workers: int,
    prefetch_factor: int,
):
    if workers <= 0:
        for image_path in image_paths:
            try:
                yield image_path, preprocess_asagi_inputs(image_path, prompt, processor), None
            except Exception as exc:
                yield image_path, None, exc
        return

    max_pending = max(1, workers * max(1, prefetch_factor))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        submit_index = 0
        for output_index, image_path in enumerate(image_paths):
            while submit_index < len(image_paths) and len(pending) < max_pending:
                pending[submit_index] = executor.submit(
                    preprocess_asagi_inputs,
                    image_paths[submit_index],
                    prompt,
                    processor,
                )
                submit_index += 1
            try:
                yield image_path, pending.pop(output_index).result(), None
            except Exception as exc:
                yield image_path, None, exc


def caption_image_asagi(
    image_path: Path,
    prompt: str,
    processor,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    preprocessed=None,
) -> str:
    if preprocessed is None:
        text_prompt, inputs = preprocess_asagi_inputs(image_path, prompt, processor)
    else:
        text_prompt, inputs = preprocessed
    inputs = {
        key: value.to(
            dtype=model.dtype if value.dtype == torch.float32 else value.dtype,
            device=model_device(model),
        )
        for key, value in inputs.items()
        if key != "token_type_ids"
    }
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    for prefix in (text_prompt, text_prompt.replace("<image>", " "), text_prompt.replace("<image>", "")):
        output_text = output_text.replace(prefix, "")
    return output_text.strip()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output = Path(args.output)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")

    block_flash_attn_for_captioning()
    from transformers import AutoModel, AutoProcessor, set_seed

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        **model_load_kwargs(args),
    )
    model.eval()
    set_seed(42)

    images = iter_images(image_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_to = Path(args.relative_to) if args.relative_to else None

    with output.open("w", encoding="utf-8") as f:
        image_iter = iter_preprocessed_asagi_inputs(
            images,
            args.prompt,
            processor,
            args.preprocess_workers,
            args.prefetch_factor,
        )
        progress = tqdm(image_iter, total=len(images), desc="Captioning ja", unit="image", dynamic_ncols=True)
        for image_path, preprocessed, preprocess_error in progress:
            try:
                if preprocess_error is not None:
                    raise preprocess_error
                text = caption_image_asagi(
                    image_path=image_path,
                    prompt=args.prompt,
                    processor=processor,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.do_sample,
                    preprocessed=preprocessed,
                )
                record = {
                    "image": jsonl_image_path(image_path, args.path_mode, relative_to),
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                tqdm.write(f"failed: {image_path}: {exc}")


if __name__ == "__main__":
    main()

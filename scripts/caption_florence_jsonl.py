#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
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
    parser.add_argument("--model", default="microsoft/Florence-2-large")
    parser.add_argument("--revision", default="21a599d414c4d928c9032694c424fb94458e3594")
    parser.add_argument("--task", default="<CAPTION>")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--square-pad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--square-pad-color", type=int, default=255)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--relative-to", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def square_pad_image(image: Image.Image, color: int) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    return ImageOps.pad(
        image,
        (size, size),
        method=Image.Resampling.BICUBIC,
        color=(color, color, color),
        centering=(0.5, 0.5),
    )


def install_florence2_compatibility_shims() -> None:
    import transformers
    from packaging.version import Version
    from transformers.configuration_utils import PretrainedConfig
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if Version(transformers.__version__) < Version("4.57"):
        return
    if getattr(PretrainedConfig, "_t2i_ja_florence2_patched", False):
        return

    original_config_getattribute = PretrainedConfig.__getattribute__

    def config_getattribute(self, key):
        if key in {"forced_bos_token_id", "forced_eos_token_id"}:
            try:
                return original_config_getattribute(self, key)
            except AttributeError:
                return None
        return original_config_getattribute(self, key)

    original_tokenizer_getattr = PreTrainedTokenizerBase.__getattr__

    def tokenizer_getattr(self, key):
        if key == "additional_special_tokens":
            return self.special_tokens_map.get("additional_special_tokens", [])
        return original_tokenizer_getattr(self, key)

    PretrainedConfig.__getattribute__ = config_getattribute
    PreTrainedTokenizerBase.__getattr__ = tokenizer_getattr
    PretrainedConfig._t2i_ja_florence2_patched = True


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


def retie_florence2_language_weights(model) -> None:
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    language_model = getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "model"):
        return
    inner_model = language_model.model
    shared = getattr(inner_model, "shared", None)
    if shared is None:
        return

    encoder = getattr(inner_model, "encoder", None)
    decoder = getattr(inner_model, "decoder", None)
    if encoder is not None and hasattr(encoder, "embed_tokens"):
        encoder.embed_tokens.weight = shared.weight
    if decoder is not None and hasattr(decoder, "embed_tokens"):
        decoder.embed_tokens.weight = shared.weight
    if hasattr(language_model, "lm_head"):
        language_model.lm_head.weight = shared.weight


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
    square_pad: bool,
    square_pad_color: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    if square_pad:
        image = square_pad_image(image, square_pad_color)
    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = move_inputs(inputs, device=device, dtype=dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=False,
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

    block_flash_attn_for_captioning()
    install_florence2_compatibility_shims()
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    retie_florence2_language_weights(model)
    model.eval()

    images = iter_images(image_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_to = Path(args.relative_to) if args.relative_to else None

    progress = tqdm(images, desc="Captioning", unit="image", dynamic_ncols=True)
    with output.open("w", encoding="utf-8") as f:
        for image_path in progress:
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
                    square_pad=args.square_pad,
                    square_pad_color=args.square_pad_color,
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

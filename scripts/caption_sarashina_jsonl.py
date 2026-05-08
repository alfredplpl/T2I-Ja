#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import MethodType

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


def patch_huggingface_hub_strict_dataclass_for_sarashina() -> None:
    try:
        import huggingface_hub.dataclasses as hub_dataclasses
    except ImportError:
        return
    if getattr(hub_dataclasses, "_t2i_ja_sarashina_patched", False):
        return

    original_type_validator = hub_dataclasses.type_validator

    def type_validator(name, value, expected_type):
        if name == "mlp_ratio" and expected_type is int and isinstance(value, float):
            return
        return original_type_validator(name, value, expected_type)

    hub_dataclasses.type_validator = type_validator
    hub_dataclasses._t2i_ja_sarashina_patched = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="MIL-UT/Asagi-2B")
    parser.add_argument("--backend", choices=["auto", "asagi", "sarashina"], default="auto")
    parser.add_argument(
        "--prompt",
        default="この画像を見て、次の指示に詳細かつ具体的に答えてください。この写真の内容について詳しく教えてください。",
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


def infer_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if "sarashina" in model_name.lower():
        return "sarashina"
    return "asagi"


def patch_sarashina_generation_cache_position(model) -> None:
    if getattr(model, "_t2i_ja_cache_position_patched", False):
        return

    original_prepare = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_with_cache_position(self, input_ids, *args, cache_position=None, **kwargs):
        if cache_position is None:
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        return original_prepare(input_ids, *args, cache_position=cache_position, **kwargs)

    model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_with_cache_position, model)
    model._t2i_ja_cache_position_patched = True


def caption_image_sarashina(
    image_path: Path,
    prompt: str,
    processor,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    image = Image.open(image_path).convert("RGB")
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(message, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model_device(model))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        use_cache=False,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def build_asagi_prompt(prompt: str) -> str:
    return (
        "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
        f"### 指示:\n<image>\n{prompt}\n\n"
        "### 応答:\n"
    )


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
) -> str:
    image = Image.open(image_path).convert("RGB")
    text_prompt = build_asagi_prompt(prompt)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    tokenized = processor.tokenizer(text_prompt, return_tensors="pt")
    inputs["input_ids"] = tokenized["input_ids"]
    inputs["attention_mask"] = tokenized["attention_mask"]
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
    backend = infer_backend(args.model, args.backend)
    if backend == "sarashina":
        patch_huggingface_hub_strict_dataclass_for_sarashina()
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, set_seed

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model_class = AutoModelForCausalLM if backend == "sarashina" else AutoModel
    model = model_class.from_pretrained(
        args.model,
        **model_load_kwargs(args),
    )
    if backend == "sarashina":
        patch_sarashina_generation_cache_position(model)
    model.eval()
    set_seed(42)

    images = iter_images(image_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_to = Path(args.relative_to) if args.relative_to else None
    caption_image = caption_image_sarashina if backend == "sarashina" else caption_image_asagi

    progress = tqdm(images, desc="Captioning ja", unit="image", dynamic_ncols=True)
    with output.open("w", encoding="utf-8") as f:
        for image_path in progress:
            try:
                text = caption_image(
                    image_path=image_path,
                    prompt=args.prompt,
                    processor=processor,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.do_sample,
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

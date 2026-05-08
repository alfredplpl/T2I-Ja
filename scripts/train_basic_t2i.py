#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import PixArtTransformer2DModel
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from t2i_ja import build_transformer, load_config
from t2i_ja.modeling import QwenTextConditioner, load_vae, sample_flow_matching_training_inputs


def round_down(value: int, multiple: int) -> int:
    return value - value % multiple


def generate_aspect_buckets(
    base_resolution: int,
    max_area: int | None = None,
    max_dim: int | None = None,
    min_dim: int = 256,
    step: int = 64,
) -> list[tuple[int, int]]:
    max_area = max_area or base_resolution * base_resolution
    max_dim = max_dim or base_resolution * 2
    buckets = {(base_resolution, base_resolution)}

    for width in range(min_dim, max_dim + 1, step):
        height = min(max_dim, max_area // width)
        height = round_down(height, step)
        if height >= min_dim:
            buckets.add((height, width))
            buckets.add((width, height))

    return sorted(buckets, key=lambda size: size[1] / size[0])


def resize_random_crop_to_bucket(image: Image.Image, bucket_size: tuple[int, int]) -> torch.Tensor:
    bucket_height, bucket_width = bucket_size
    width, height = image.size
    scale = max(bucket_width / width, bucket_height / height)
    resized_width = math.ceil(width * scale)
    resized_height = math.ceil(height * scale)
    image = TF.resize(
        image,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
    )
    top = 0 if resized_height == bucket_height else random.randint(0, resized_height - bucket_height)
    left = 0 if resized_width == bucket_width else random.randint(0, resized_width - bucket_width)
    image = TF.crop(image, top, left, bucket_height, bucket_width)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5], [0.5])


class JsonlImageTextDataset(Dataset):
    def __init__(self, path: str, resolution: int, bucket_config: dict | None = None) -> None:
        self.samples = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.bucket_config = bucket_config or {}
        self.buckets: list[tuple[int, int]] = []
        self.sample_buckets: list[int] = []
        self.bucket_to_indices: dict[int, list[int]] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if self.bucket_config.get("enabled", False):
            self._assign_buckets(resolution)

    def _assign_buckets(self, resolution: int) -> None:
        self.buckets = generate_aspect_buckets(
            base_resolution=resolution,
            max_area=self.bucket_config.get("max_area"),
            max_dim=self.bucket_config.get("max_dim"),
            min_dim=int(self.bucket_config.get("min_dim", 256)),
            step=int(self.bucket_config.get("step", 64)),
        )
        bucket_aspects = [width / height for height, width in self.buckets]
        max_aspect_error = self.bucket_config.get("max_aspect_error")
        kept_samples = []
        for sample in self.samples:
            with Image.open(sample["image"]) as image:
                width, height = image.size
            aspect = width / height
            bucket_index, aspect_error = min(
                enumerate(abs(bucket_aspect - aspect) for bucket_aspect in bucket_aspects),
                key=lambda item: item[1],
            )
            if max_aspect_error is not None and aspect_error > float(max_aspect_error):
                continue
            kept_samples.append(sample)
            self.sample_buckets.append(bucket_index)
            self.bucket_to_indices.setdefault(bucket_index, []).append(len(kept_samples) - 1)
        self.samples = kept_samples
        if not self.samples:
            raise ValueError("No training samples remain after aspect ratio bucket assignment.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample["image"]).convert("RGB")
        if self.bucket_config.get("enabled", False):
            bucket = self.buckets[self.sample_buckets[index]]
            pixel_values = resize_random_crop_to_bucket(image, bucket)
        else:
            pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "text": sample["text"]}


class AspectRatioBucketBatchSampler(BatchSampler):
    def __init__(self, dataset: JsonlImageTextDataset, batch_size: int, drop_last: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        for indices in self.dataset.bucket_to_indices.values():
            shuffled = list(indices)
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        count = 0
        for indices in self.dataset.bucket_to_indices.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += math.ceil(len(indices) / self.batch_size)
        return count


def build_optimizer(parameters, train_config: dict):
    optimizer_name = train_config.get("optimizer", "adamw8bit")
    learning_rate = float(train_config["learning_rate"])
    if optimizer_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("optimizer=adamw8bit requires bitsandbytes. Run `uv sync`.") from exc

        return bnb.optim.AdamW8bit(parameters, lr=learning_rate)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def first_learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def load_or_build_transformer(config, dtype: torch.dtype, device: torch.device, resume_transformer: str | None):
    if resume_transformer is None:
        transformer = build_transformer(config)
    else:
        transformer = PixArtTransformer2DModel.from_pretrained(resume_transformer, torch_dtype=dtype)
    return transformer.to(device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-transformer", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config.train
    dtype_name = args.dtype or train_config.get("mixed_precision", "bf16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(train_config["batch_size"])
    dataset = JsonlImageTextDataset(
        args.data,
        int(config.image["resolution"]),
        bucket_config=train_config.get("aspect_ratio_bucketing"),
    )
    if train_config.get("aspect_ratio_bucketing", {}).get("enabled", False):
        bucket_config = train_config["aspect_ratio_bucketing"]
        loader = DataLoader(
            dataset,
            batch_sampler=AspectRatioBucketBatchSampler(
                dataset,
                batch_size=batch_size,
                drop_last=bool(bucket_config.get("drop_last", False)),
            ),
            num_workers=int(train_config["num_workers"]),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(train_config["num_workers"]),
        )

    vae = load_vae(config, dtype=dtype, device=device)
    vae.eval().requires_grad_(False)
    text = QwenTextConditioner(config, dtype=dtype, device=device)
    transformer = load_or_build_transformer(
        config,
        dtype=dtype,
        device=device,
        resume_transformer=args.resume_transformer,
    )
    if train_config.get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
    optimizer = build_optimizer(transformer.parameters(), train_config)
    autocast_enabled = dtype is not torch.float32

    global_step = 0
    accumulation = int(train_config["gradient_accumulation_steps"])
    total_steps = args.max_steps or (int(train_config["epochs"]) * len(loader))
    progress = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)
    transformer.train()
    try:
        for epoch in range(int(train_config["epochs"])):
            if global_step >= total_steps:
                break
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                prompts = list(batch["text"])
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    condition = text(prompts, device)
                    noisy_latents, timesteps, target = sample_flow_matching_training_inputs(
                        latents,
                        int(config.scheduler["train_timesteps"]),
                    )

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    prediction = transformer(
                        noisy_latents,
                        encoder_hidden_states=condition.hidden_states,
                        encoder_attention_mask=condition.attention_mask,
                        timestep=timesteps,
                    ).sample
                loss = F.mse_loss(prediction.float(), target.float()) / accumulation
                loss.backward()

                optimizer_step = (global_step + 1) % accumulation == 0
                if optimizer_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress.update(1)
                progress.set_postfix(
                    epoch=epoch + 1,
                    loss=f"{loss.detach().float().item() * accumulation:.4f}",
                    lr=f"{first_learning_rate(optimizer):.2e}",
                    opt_step=int(optimizer_step),
                )
                if global_step % int(train_config["save_every_steps"]) == 0:
                    transformer.save_pretrained(output_dir / f"transformer-{global_step}")
                if global_step >= total_steps:
                    transformer.save_pretrained(output_dir / "transformer-final")
                    return
    finally:
        progress.close()

    transformer.save_pretrained(output_dir / "transformer-final")


if __name__ == "__main__":
    main()

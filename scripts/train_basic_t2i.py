#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from t2i_ja import build_transformer, load_config
from t2i_ja.modeling import QwenTextConditioner, build_training_scheduler, load_qwen_image_vae


class JsonlImageTextDataset(Dataset):
    def __init__(self, path: str, resolution: int) -> None:
        self.samples = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample["image"]).convert("RGB")
        return {"pixel_values": self.transform(image), "text": sample["text"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
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

    dataset = JsonlImageTextDataset(args.data, int(config.image["resolution"]))
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config["num_workers"]),
    )

    vae = load_qwen_image_vae(config, dtype=dtype, device=device)
    vae.eval().requires_grad_(False)
    text = QwenTextConditioner(config, dtype=dtype, device=device)
    transformer = build_transformer(config).to(device=device)
    scheduler = build_training_scheduler(config)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=float(train_config["learning_rate"]))
    autocast_enabled = dtype is not torch.float32

    global_step = 0
    accumulation = int(train_config["gradient_accumulation_steps"])
    transformer.train()
    for _epoch in range(int(train_config["epochs"])):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = list(batch["text"])
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                condition = text(prompts, device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                prediction = transformer(
                    noisy_latents,
                    encoder_hidden_states=condition.hidden_states,
                    encoder_attention_mask=condition.attention_mask,
                    timestep=timesteps,
                ).sample
            loss = F.mse_loss(prediction.float(), noise.float()) / accumulation
            loss.backward()

            if (global_step + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % int(train_config["save_every_steps"]) == 0:
                transformer.save_pretrained(output_dir / f"transformer-{global_step}")
            if args.max_steps is not None and global_step >= args.max_steps:
                transformer.save_pretrained(output_dir / "transformer-final")
                return

    transformer.save_pretrained(output_dir / "transformer-final")


if __name__ == "__main__":
    main()

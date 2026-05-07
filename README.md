# T2I-Ja

Minimal Text-to-Image training/inference scaffold using:

- VAE: `Qwen/Qwen-Image-2512`, subfolder `vae`
- Text encoder: `Qwen/Qwen3.5-2B-Base`
- DiT: Diffusers `PixArtTransformer2DModel`, configured as an approximately 2B-class transformer
- Generation: Diffusers `PixArtSigmaPipeline` with Qwen prompt embeddings passed through `prompt_embeds`

This repository starts from randomly initialized PixArt weights. It is a basic T2I implementation for training or smoke-testing the wiring, not a pretrained image generator.

The standard `PixArtSigmaPipeline` tokenizer/text encoder path is T5-specific. This project keeps the pipeline itself, but encodes prompts with Qwen first and calls the pipeline with `prompt_embeds`, `prompt_attention_mask`, `negative_prompt_embeds`, and `negative_prompt_attention_mask`.

The default DiT config uses 40 layers with hidden size 2048 (`16 heads x 128 dim`) and Qwen text embeddings with 2048 channels. This is intended as a roughly 2B-class DiT setting, not a PixArt pretrained checkpoint-compatible shape.

## Setup With uv

```bash
uv sync
```

`Qwen/Qwen-Image-2512` uses `AutoencoderKLQwenImage`, so use a recent Diffusers build that includes Qwen Image support.

This project is configured for PyTorch CUDA 12.8 wheels on Linux and Windows via uv:

```bash
uv sync
```

The CUDA wheel index is `https://download.pytorch.org/whl/cu128`. On macOS, uv falls back to PyPI because CUDA wheels are not available there.

FlashAttention is installed from the GitHub release asset for v2.8.3, not built from PyPI:

```text
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

This wheel is for Linux x86_64, Python 3.12, Torch 2.8, CUDA 12.x, and the CXX11 ABI PyTorch build. The project pins `torch==2.8.0` and `torchvision==0.23.0` so the FlashAttention wheel and PyTorch ABI match. For a different Python/Torch/ABI combination, choose the matching wheel from the Assets section of the v2.8.3 release and update the `flash-attn @ ...` URL in `pyproject.toml`.

The Qwen text encoder is configured with `attn_implementation: flash_attention_2` in `configs/basic_t2i_qwen_pixart.yaml`.

## Train

Create a JSONL file with one sample per line:

```json
{"image": "/path/to/image.png", "text": "a short caption"}
```

Then run:

```bash
uv run t2i-train \
  --config configs/basic_t2i_qwen_pixart.yaml \
  --data train.jsonl \
  --output-dir outputs/basic-t2i
```

Only the DiT/PixArt transformer is trained. The Qwen Image VAE and Qwen text encoder are frozen.

## Infer

After training:

```bash
uv run t2i-infer \
  --config configs/basic_t2i_qwen_pixart.yaml \
  --checkpoint outputs/basic-t2i/transformer-final \
  --prompt "東京の夜景、水彩画" \
  --output outputs/sample.png
```

## Export Pipeline

To write a Diffusers pipeline directory:

```bash
uv run t2i-export-pipeline \
  --config configs/basic_t2i_qwen_pixart.yaml \
  --checkpoint outputs/basic-t2i/transformer-final \
  --output-dir outputs/basic-t2i/pixart-sigma-pipeline
```

The exported pipeline still needs externally computed Qwen prompt embeddings at inference time because `PixArtSigmaPipeline` does not natively encode Qwen text prompts.

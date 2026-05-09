# T2I-Ja

Minimal Text-to-Image training/inference scaffold using:

- VAE: `black-forest-labs/FLUX.2-klein-base-4B`, subfolder `vae`
- Text encoder: `Qwen/Qwen3.5-2B-Base`
- DiT: Diffusers `PixArtTransformer2DModel`, configured as an approximately 2B-class transformer
- Generation: Diffusers `PixArtSigmaPipeline` with Qwen prompt embeddings passed through `prompt_embeds`
- Objective: Flow Matching velocity prediction with `FlowMatchEulerDiscreteScheduler`

This repository starts from randomly initialized PixArt weights. It is a basic T2I implementation for training or smoke-testing the wiring, not a pretrained image generator.

The standard `PixArtSigmaPipeline` tokenizer/text encoder path is T5-specific. This project keeps the pipeline itself, but encodes prompts with Qwen first and calls the pipeline with `prompt_embeds`, `prompt_attention_mask`, `negative_prompt_embeds`, and `negative_prompt_attention_mask`.

The default DiT config uses 40 layers with hidden size 2048 (`16 heads x 128 dim`) and Qwen text embeddings with 2048 channels. This is intended as a roughly 2B-class DiT setting, not a PixArt pretrained checkpoint-compatible shape.

## Setup With uv

```bash
uv sync
```

The FLUX.2 klein VAE uses `AutoencoderKLFlux2`, so use Diffusers 0.37.0 or newer. `black-forest-labs/FLUX.2-klein-base-4B` is Apache-2.0 licensed on Hugging Face.

This project is configured for PyTorch CUDA 12.8 wheels on Linux and Windows via uv:

```bash
uv sync
```

The CUDA wheel index is `https://download.pytorch.org/whl/cu128`. On macOS, uv falls back to PyPI because CUDA wheels are not available there.

FlashAttention is optional. Install it only for Qwen/PixArt training or inference runs that need `attn_implementation: flash_attention_2`:

```bash
uv sync --group flash-attn
```

FlashAttention is installed from the GitHub release asset for v2.8.3, not built from PyPI:

```text
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

This wheel is for Linux x86_64, Python 3.12, Torch 2.8, CUDA 12.x, and the CXX11 ABI PyTorch build. The project pins `torch==2.8.0` and `torchvision==0.23.0` so the FlashAttention wheel and PyTorch ABI match. For a different Python/Torch/ABI combination, choose the matching wheel from the Assets section of the v2.8.3 release and update the `flash-attn @ ...` URL in `pyproject.toml`.

The Qwen text encoder is configured with `attn_implementation: flash_attention_2` in `configs/basic_t2i_qwen_pixart.yaml`.

## Caption Images

Generate a training JSONL from an image directory with Florence-2:

```bash
uv run --script scripts/t2i_caption.py \
  --image-dir /path/to/images \
  --output train.jsonl \
  --recursive \
  --overwrite
```

The command uses `microsoft/Florence-2-base` and `<CAPTION>` by default. Each output line matches the training format:

```json
{"image": "/path/to/image.png", "text": "a detailed Florence-2 caption"}
```

Useful options:

- `--image-dir`: directory containing images
- `--output`: JSONL output path
- `--recursive`: include images in nested directories
- `--limit`: process only the first N images
- `--overwrite`: replace an existing output file
- `--device`: device for Florence-2, default `cuda`
- `--dtype`: `auto`, `fp32`, `fp16`, or `bf16`; default `auto`
- `--task`: Florence task prompt, default `<CAPTION>`
- `--attn-implementation`: attention backend for Florence-2, default `eager`
- `--revision`: optional Florence model revision
- `--max-new-tokens`: generation length limit, default `1024`
- `--num-beams`: beam count, default `3`
- `--square-pad` / `--no-square-pad`: pad non-square images to square before Florence-2, default enabled
- `--square-pad-color`: RGB padding color value, default `255`
- `--continue-on-error`: skip unreadable or failed images

Progress is shown with `tqdm`, including elapsed time, images per second, and estimated remaining time. Per-image filenames and generated captions are not printed during normal runs.

By default, image paths are absolute. To write portable relative paths:

```bash
uv run --script scripts/t2i_caption.py \
  --image-dir /data/images \
  --output train.jsonl \
  --recursive \
  --path-mode relative \
  --relative-to /data \
  --overwrite
```

This writes paths like `images/example.png`.

The script keeps `trust_remote_code=True` because the Microsoft Florence-2 repository still relies on custom processor/model code for this workflow. It also installs a small compatibility shim for newer `transformers` versions where the Florence-2 remote config/tokenizer code can otherwise fail with missing `forced_bos_token_id` or `additional_special_tokens` attributes. Florence-2 defaults to `--attn-implementation eager` because the remote model class does not expose the SDPA support flags expected by newer `transformers` releases. Generation runs with `use_cache=False` to avoid the newer `EncoderDecoderCache` API that the Florence-2 remote generation code does not support.

`scripts/t2i_caption.py` automatically re-runs the captioner with the project's existing Python environment plus `transformers==4.51.3` as an overlay. It uses `--no-project`, so captioning does not try to sync the main project dependencies or download the optional `flash-attn` wheel. This keeps Florence-2's older remote code compatible without downgrading the main T2I environment used by Qwen/PixArt.

The runner also uses uv's offline cache for the overlay environment to avoid touching the network during captioning. If the Florence caption environment has never been cached on the machine, prime it once while online:

```bash
uv run --no-project \
  --python .venv/bin/python \
  --with transformers==4.51.3 \
  --with pillow \
  --with timm \
  --with einops \
  --with tqdm \
  python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
```

## Japanese Caption Images

Generate Japanese captions with `MIL-UT/Asagi-2B`:

```bash
uv run t2i-caption-ja \
  --image-dir /path/to/images \
  --output train_ja.jsonl \
  --recursive \
  --overwrite
```

The default prompt is:

```text
簡潔に説明してください。
```

The output format is the same training JSONL format:

```json
{"image": "/path/to/image.webp", "text": "水槽が木製の机の上に置かれている写真です。..."}
```

Useful options:

- `--prompt`: Japanese instruction prompt
- `--model`: model id, default `MIL-UT/Asagi-2B`
- `--device-map`: model placement, default `cuda`
- `--dtype`: `auto`, `fp32`, `fp16`, or `bf16`; default `auto`
- `--attn-implementation`: optional attention backend override for model loading
- `--max-new-tokens`: generation length limit, default `256`
- `--temperature`: default `0.7`
- `--top-p`: default `0.95`
- `--repetition-penalty`: default `1.2`
- `--do-sample` / `--no-do-sample`: sampling mode, default enabled
- `--preprocess-workers`: number of background threads used for Asagi image loading and processor/tokenizer preprocessing, default `0`
- `--prefetch-factor`: queued images per preprocessing worker, default `2`
- `--recursive`, `--limit`, `--path-mode`, `--relative-to`, `--overwrite`, `--continue-on-error`: same behavior as the Florence caption command

Progress is shown with `tqdm`, and per-image filenames and generated captions are not printed during normal runs.

The Japanese caption runner overlays `transformers==4.45.1` and `tokenizers==0.20.3` only for this command, matching the Asagi model card while leaving the main Qwen/PixArt `torch` and `torchvision` environment unchanged.

## Train

Use a JSONL file with one sample per line:

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

Only the DiT/PixArt transformer is trained. The FLUX.2 klein VAE and Qwen text encoder are frozen. Training uses Flow Matching: it samples `sigma`, builds `x_t = (1 - sigma) * x_0 + sigma * noise`, and trains the transformer to predict `noise - x_0`.

Training uses aspect ratio bucketing by default. Images are assigned to the closest bucket, resized while preserving aspect ratio, then randomly cropped to the bucket resolution. Each batch contains images from one bucket, so `batch_size > 1` still works without padding.

The default buckets follow the NovelAI-style setup with a `512 * 768` pixel budget, maximum dimension `1024`, minimum dimension `256`, and `64` pixel steps:

```yaml
train:
  seed: 42
  lr_scheduler: constant
  aspect_ratio_bucketing:
    enabled: true
    max_area: 393216
    max_dim: 1024
    min_dim: 256
    step: 64
    drop_last: false
```

Set `enabled: false` to return to the previous square `Resize + CenterCrop` preprocessing.

Set `train.seed` or pass `--seed` to make sample ordering, bucket shuffling, random crops, model initialization, and Flow Matching noise sampling reproducible. The CLI value overrides the config value.

Use `lr_scheduler: constant_with_warmup` with `warmup_steps` to linearly ramp from zero to `learning_rate`, then keep it constant. Warmup advances on optimizer steps, not gradient accumulation micro steps.

Trackio logging can be enabled from the config or CLI. It logs `train/loss`, `train/lr`, `train/epoch`, and whether the current step performed an optimizer update:

```yaml
train:
  trackio:
    enabled: true
    project: t2i-ja
    log_every_steps: 1
    auto_log_gpu: false
```

```bash
uv run t2i-train \
  --config configs/stage_256.yaml \
  --data train.jsonl \
  --output-dir outputs/basic-t2i-256 \
  --trackio \
  --trackio-name stage-256
```

To train progressively, run separate stages and resume only the transformer weights while changing the config resolution and bucket budget. The optimizer is intentionally re-created for each stage.

```bash
uv run t2i-train \
  --config configs/stage_256.yaml \
  --data train.jsonl \
  --output-dir outputs/basic-t2i-256

uv run t2i-train \
  --config configs/stage_512.yaml \
  --data train.jsonl \
  --output-dir outputs/basic-t2i-512 \
  --resume-transformer outputs/basic-t2i-256/transformer-final

uv run t2i-train \
  --config configs/stage_1024.yaml \
  --data train.jsonl \
  --output-dir outputs/basic-t2i-1024 \
  --resume-transformer outputs/basic-t2i-512/transformer-final
```

## Infer

After training:

```bash
uv run t2i-infer \
  --config configs/basic_t2i_qwen_pixart.yaml \
  --checkpoint outputs/basic-t2i/transformer-final \
  --prompt "東京の夜景、水彩画" \
  --height 768 \
  --width 512 \
  --output outputs/sample.png
```

Useful options:

- `--config`: YAML config used to build the VAE, text encoder, scheduler, and transformer shape. Use the same architecture config family as training.
- `--checkpoint`: transformer checkpoint directory, such as `outputs/basic-t2i/transformer-final` or `outputs/basic-t2i/transformer-4000`.
- `--prompt`: prompt text encoded by the Qwen text encoder.
- `--output`: output image path. Parent directories are created automatically.
- `--height` / `--width`: output size in pixels. If omitted, both default to `image.resolution` from the config. Use values compatible with the VAE downsampling and transformer patch size; multiples of 16 are the practical default.
- `--steps`: number of Flow Matching Euler denoising steps, default `30`. Lower values are faster; higher values cost more time.
- `--guidance-scale`: classifier-free guidance scale, default `4.5`.
- `--seed`: random seed for the initial latent noise, default `0`.
- `--device`: torch device, default `cuda`.
- `--dtype`: `bf16`, `fp16`, or `fp32`, default `bf16`.

`t2i-infer` currently uses an empty negative prompt internally; there is no CLI option for negative prompts yet.

## Export Pipeline

To write a Diffusers pipeline directory:

```bash
uv run t2i-export-pipeline \
  --config configs/basic_t2i_qwen_pixart.yaml \
  --checkpoint outputs/basic-t2i/transformer-final \
  --output-dir outputs/basic-t2i/pixart-sigma-pipeline
```

The exported pipeline still needs externally computed Qwen prompt embeddings at inference time because `PixArtSigmaPipeline` does not natively encode Qwen text prompts.

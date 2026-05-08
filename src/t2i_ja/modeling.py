from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel
from transformers import AutoModel, AutoTokenizer

from .config import T2IConfig


def load_vae(config: T2IConfig, dtype: torch.dtype, device: torch.device):
    vae_class_name = config.models.get("vae_class", "AutoencoderKLFlux2")
    try:
        vae_class = getattr(diffusers, vae_class_name)
    except AttributeError as exc:
        raise ImportError(
            f"{vae_class_name} is required. Install a recent diffusers build that includes it."
        ) from exc

    vae = vae_class.from_pretrained(
        config.models["vae_name"],
        subfolder=config.models.get("vae_subfolder", "vae"),
        torch_dtype=dtype,
    ).to(device)
    if not hasattr(vae.config, "scaling_factor"):
        vae.register_to_config(scaling_factor=float(config.models.get("vae_scaling_factor", 1.0)))
    return vae


def build_transformer(config: T2IConfig) -> PixArtTransformer2DModel:
    return PixArtTransformer2DModel(
        sample_size=config.transformer["sample_size"],
        patch_size=config.transformer["patch_size"],
        in_channels=config.transformer["in_channels"],
        out_channels=config.transformer["out_channels"],
        num_layers=config.transformer["num_layers"],
        num_attention_heads=config.transformer["num_attention_heads"],
        attention_head_dim=config.transformer["attention_head_dim"],
        cross_attention_dim=config.transformer["cross_attention_dim"],
        caption_channels=config.transformer["caption_channels"],
        use_additional_conditions=False,
    )


@dataclass
class TextCondition:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class QwenTextConditioner:
    def __init__(self, config: T2IConfig, dtype: torch.dtype, device: torch.device) -> None:
        self.max_length = int(config.text["max_length"])
        name = config.models["text_encoder_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if config.text.get("attn_implementation"):
            model_kwargs["attn_implementation"] = config.text["attn_implementation"]
        try:
            self.text_encoder = AutoModel.from_pretrained(
                name,
                **model_kwargs,
            ).to(device)
        except Exception:
            if "attn_implementation" not in model_kwargs:
                raise
            failed_attn = model_kwargs.pop("attn_implementation")
            warnings.warn(
                f"Failed to load text encoder with attn_implementation={failed_attn!r}; "
                "falling back to the model default attention implementation.",
                stacklevel=2,
            )
            self.text_encoder = AutoModel.from_pretrained(
                name,
                **model_kwargs,
            ).to(device)
        self.text_encoder.eval()
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts: list[str], device: torch.device) -> TextCondition:
        tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        output = self.text_encoder(**tokens)
        hidden_states = output.last_hidden_state
        return TextCondition(hidden_states=hidden_states, attention_mask=tokens.attention_mask)


def build_pixart_sigma_pipeline(
    config: T2IConfig,
    transformer: PixArtTransformer2DModel,
    dtype: torch.dtype,
    device: torch.device,
) -> PixArtSigmaPipeline:
    vae = load_vae(config, dtype=dtype, device=device)
    vae.eval().requires_grad_(False)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(config.scheduler["train_timesteps"]),
        shift=float(config.scheduler.get("shift", 1.0)),
    )
    pipe = PixArtSigmaPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer.to(device=device, dtype=dtype),
        scheduler=scheduler,
    )
    return pipe.to(device)


def encode_qwen_prompt(
    config: T2IConfig,
    prompts: str | list[str],
    dtype: torch.dtype,
    device: torch.device,
    negative_prompts: str | list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_list = [prompts] if isinstance(prompts, str) else prompts
    if negative_prompts is None:
        negative_list = [""] * len(prompt_list)
    elif isinstance(negative_prompts, str):
        negative_list = [negative_prompts] * len(prompt_list)
    else:
        negative_list = negative_prompts

    conditioner = QwenTextConditioner(config, dtype=dtype, device=device)
    prompt = conditioner(prompt_list, device)
    negative = conditioner(negative_list, device)
    return (
        prompt.hidden_states,
        prompt.attention_mask,
        negative.hidden_states,
        negative.attention_mask,
    )


def sample_flow_matching_training_inputs(
    latents: torch.Tensor,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    sigmas = torch.rand((latents.shape[0],), device=latents.device, dtype=latents.dtype)
    broadcast_sigmas = sigmas.view(-1, *([1] * (latents.ndim - 1)))
    noisy_latents = (1.0 - broadcast_sigmas) * latents + broadcast_sigmas * noise
    target = noise - latents
    timesteps = (sigmas * num_train_timesteps).long()
    return noisy_latents, timesteps, target

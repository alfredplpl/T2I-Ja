from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class T2IConfig:
    raw: dict[str, Any]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def image(self) -> dict[str, Any]:
        return self.raw["image"]

    @property
    def text(self) -> dict[str, Any]:
        return self.raw["text"]

    @property
    def transformer(self) -> dict[str, Any]:
        return self.raw["transformer"]

    @property
    def scheduler(self) -> dict[str, Any]:
        return self.raw["scheduler"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]


def load_config(path: str | Path) -> T2IConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return T2IConfig(yaml.safe_load(f))

#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run the Japanese caption environment.")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "caption_sarashina_jsonl.py"
    env = os.environ.copy()
    command = [
        uv,
        "run",
        "--no-project",
        "--python",
        str(repo_root / ".venv" / "bin" / "python"),
        "--with",
        "transformers==4.45.1",
        "--with",
        "tokenizers==0.20.3",
        "python",
        str(script),
        *sys.argv[1:],
    ]
    os.execvpe(uv, command, env)


if __name__ == "__main__":
    main()

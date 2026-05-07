from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    if os.environ.get("T2I_JA_CAPTION_ISOLATED") == "1":
        from ._script_path import add_repo_root_to_path

        add_repo_root_to_path()
        from scripts.caption_florence_jsonl import main as caption_main

        caption_main()
        return

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run t2i-caption in its isolated Florence-2 environment.")

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "caption_florence_jsonl.py"
    env = os.environ.copy()
    env["T2I_JA_CAPTION_ISOLATED"] = "1"
    command = [
        uv,
        "run",
        "--isolated",
        "--python",
        "3.12",
        "--index",
        "https://download.pytorch.org/whl/cu128",
        "--with",
        "torch==2.8.0",
        "--with",
        "torchvision==0.23.0",
        "--with",
        "transformers==4.51.3",
        "--with",
        "pillow",
        "--with",
        "timm",
        "--with",
        "einops",
        "python",
        str(script),
        *sys.argv[1:],
    ]
    os.execvpe(uv, command, env)


if __name__ == "__main__":
    main()

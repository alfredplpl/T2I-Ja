from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run t2i-caption-ja.")

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "caption_sarashina_jsonl.py"
    env = os.environ.copy()
    command = [
        uv,
        "run",
        "--no-project",
        "--offline",
        "--python",
        str(repo_root / ".venv" / "bin" / "python"),
        "--with",
        "transformers>=4.57.1",
        "--with",
        "pillow",
        "--with",
        "protobuf",
        "--with",
        "sentencepiece",
        "--with",
        "accelerate",
        "--with",
        "tqdm",
        "python",
        str(script),
        *sys.argv[1:],
    ]
    os.execvpe(uv, command, env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the existing AEGIS Llama-Guard judge on attack outputs.

This is a thin wrapper so the judge stays exactly the AEGIS implementation.
Run it in a separate process from the attack model to avoid GPU memory overlap.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "aegis" / "judge.py"
DEFAULT_GUARD = "/data/lar/models/Llama-Guard-3-8B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--response-key", default="qwen2_audio_response")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--model", default=DEFAULT_GUARD)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    cmd = [
        sys.executable, str(JUDGE),
        "--input", str(Path(args.input).resolve()),
        "--out", str(Path(args.out).resolve()),
        "--response-key", args.response_key,
        "--prompt-key", args.prompt_key,
        "--model", args.model,
        "--device", args.device,
        "--max-new-tokens", str(args.max_new_tokens),
        "--local-files-only",
    ]
    print("[judge]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT / "aegis"))


if __name__ == "__main__":
    main()

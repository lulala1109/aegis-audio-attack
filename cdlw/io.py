"""Small I/O helpers for attack runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "sample"


def snr_db(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    noise = (adversarial - original).float()
    signal_power = torch.mean(original.float() ** 2)
    noise_power = torch.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * torch.log10(signal_power / noise_power))


def save_wav_pair(
    waveform: torch.Tensor,
    out_dir: str | Path,
    sample_rate: int,
    prefix: str = "adversarial",
) -> tuple[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
    float_path = out_dir / f"{prefix}.float.wav"
    pcm_path = out_dir / f"{prefix}.pcm16.wav"
    sf.write(float_path, array, sample_rate, subtype="FLOAT")
    sf.write(pcm_path, array, sample_rate, subtype="PCM_16")
    return str(float_path), str(pcm_path)


def tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().flatten()]

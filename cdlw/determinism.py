"""Determinism setup for waveform PGD experiments.

The Qwen2-Audio attack path uses bf16 CUDA kernels and a PyTorch STFT.  Without
explicit settings, repeating the same backward pass can produce different
gradients, which sign-PGD then amplifies across steps.  These settings reduce
that nondeterminism as far as the installed PyTorch/CUDA stack allows.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_determinism(seed: int, strict: bool = False) -> dict:
    """Seed all RNGs and enable deterministic CUDA/cuBLAS behaviour.

    ``strict=True`` turns ``warn_only`` off, so unsupported nondeterministic
    operators raise instead of only warning.  The default is warn-only because
    bf16 attention/STFT kernels vary across PyTorch builds.
    """
    # Must be set before the first cuBLAS call.  :4096:8 is the documented
    # deterministic workspace for CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    seed_everything(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 is a source of run-to-run variation in FP32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        deterministic_algorithms = True
    except Exception:
        deterministic_algorithms = False

    return {
        "seed": int(seed),
        "strict": bool(strict),
        "deterministic_algorithms": deterministic_algorithms,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Dedicated generator so delta init does not depend on global RNG state."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator

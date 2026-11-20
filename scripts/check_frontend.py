#!/usr/bin/env python3
"""Validate the differentiable frontend, layer readout, and input gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdlw.bridge import aegis_common, configure_offline  # noqa: E402
from cdlw.io import load_jsonl, write_json  # noqa: E402
from cdlw.model import QwenAudioAttackModel  # noqa: E402
from cdlw.objectives import internal_loss  # noqa: E402

DEFAULT_MODEL = (
    "/data/lar/hf-cache/hub/"
    "models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/"
    "0a095220c30b7b31434169c3086508ef3ea5bf0a"
)
DEFAULT_HF_HOME = "/data/lar/hf-cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="runs/frontend_check.json")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", default="native", choices=["native", "safety"])
    parser.add_argument("--target", default="Sure, here is")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    args = parser.parse_args()

    configure_offline(args.hf_home)
    if args.audio:
        audio_path = args.audio
    else:
        if not args.manifest:
            parser.error("--audio or --manifest is required")
        rows = load_jsonl(args.manifest)
        common = aegis_common()
        audio_path = common.resolve_audio(common.audio_path_of(rows[0]))
    print(f"[INFO] audio={audio_path}")
    print(f"[INFO] model={args.model_path}")

    model = QwenAudioAttackModel(
        model_path=args.model_path,
        device=args.device,
        prompt_mode=args.prompt_mode,
        hf_home=args.hf_home,
        max_seconds=args.max_seconds,
    )
    waveform = model.load_wav(audio_path)
    layout = model.build_layout(waveform, args.target)
    print(f"[INFO] layers={model.num_layers} prompt_len={layout.prompt_len}")

    differentiable = model.frontend(waveform)
    official = layout.official_input_features.to(differentiable.device)
    diff = (official - differentiable).abs()
    rel_l2 = float(torch.norm(official - differentiable) / torch.norm(official))
    feature_report = {
        "shape_official": list(official.shape),
        "shape_differentiable": list(differentiable.shape),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_l2": rel_l2,
    }
    print(f"[frontend] {json.dumps(feature_report)}")

    layers = [15, 19, 25, 31]
    waveform = waveform.detach().clone().requires_grad_(True)
    pack = model.forward_attack(waveform, layout, layers)
    reference = model.reference_logprob(0.1)
    internal = sum(internal_loss(pack.refusal_logprobs[layer], reference) for layer in layers)
    loss = pack.base_loss + 0.1 * internal
    grad = torch.autograd.grad(loss, waveform)[0]
    parameter_grads = [p.grad for p in model.model.parameters() if p.grad is not None]

    readout = {
        str(layer): {
            "hidden_shape": list(pack.layer_states[layer].shape),
            "refusal_prob": float(pack.refusal_logprobs[layer].exp().item()),
        }
        for layer in layers
    }
    report = {
        "audio": audio_path,
        "model": args.model_path,
        "prompt_mode": args.prompt_mode,
        "num_layers": model.num_layers,
        "prompt_len": layout.prompt_len,
        "target_ids": [int(x) for x in layout.labels[0, layout.prompt_len:].tolist()],
        "frontend": feature_report,
        "base_loss": float(pack.base_loss.item()),
        "internal_loss": float(internal.item()),
        "readout": readout,
        "gradient": {
            "finite": bool(torch.isfinite(grad).all().item()),
            "nonzero": bool((grad != 0).any().item()),
            "norm": float(grad.norm().item()),
            "max_abs": float(grad.abs().max().item()),
            "parameter_grads_present": len(parameter_grads),
        },
    }
    write_json(args.out, report)
    print(f"[readout] {json.dumps(readout)}")
    print(f"[gradient] {json.dumps(report['gradient'])}")
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()

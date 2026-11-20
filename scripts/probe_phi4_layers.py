#!/usr/bin/env python3
"""Probe per-layer refusal probabilities on Phi-4-multimodal (logit lens).

Runs the standard attack layout (native prompt + target) on a handful of JBB
samples and records, for every decoder layer, the summed first-token refusal
probability read through the final-norm + lm_head logit lens. This shows where
the refusal signal actually lives on Phi-4, so the layer prior (currently the
Qwen-tuned 19-31) can be replaced with a Phi-4-specific one.

Usage:
  python scripts/probe_phi4_layers.py \
      --manifest runs/original_jbb/jbb_safe_lar.jsonl \
      --limit 10 --device cuda:0 --out runs/phi4_layer_probe.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdlw.bridge import configure_offline  # noqa: E402
from cdlw.io import load_jsonl, write_json  # noqa: E402
from cdlw.model import Phi4AudioAttackModel  # noqa: E402

DEFAULT_MODEL = "/data/lar/models/Phi-4-multimodal-instruct"
DEFAULT_HF_HOME = "/data/lar/hf-cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", default="native", choices=["native", "safety"])
    parser.add_argument("--target", default="Sure, here is")
    parser.add_argument("--out", default="runs/phi4_layer_probe.json")
    args = parser.parse_args()

    configure_offline(args.hf_home)
    print(f"[INFO] loading {args.model_path} on {args.device}")
    model = Phi4AudioAttackModel(
        model_path=args.model_path,
        device=args.device,
        prompt_mode=args.prompt_mode,
        hf_home=args.hf_home,
        max_seconds=20.0,
    )
    print(f"[INFO] layers={model.num_layers}")

    rows = load_jsonl(args.manifest)
    step = max(1, len(rows) // args.limit)
    rows = rows[::step][: args.limit]
    print(f"[INFO] probing {len(rows)} samples (every {step}th)")

    all_layers = list(range(model.num_layers))
    per_sample = []
    for row in rows:
        sample_id = row.get("id") or row.get("sample_id") or "?"
        audio_path = row.get("local_audio")
        if not audio_path or not Path(audio_path).exists():
            print(f"[SKIP] {sample_id}: missing {audio_path}")
            continue
        waveform = model.load_wav(audio_path)
        layout = model.build_layout(waveform, args.target)
        pack = model.forward_attack(waveform, layout, all_layers)
        probs = {
            str(layer): float(pack.refusal_logprobs[layer].exp().item())
            for layer in all_layers
        }
        per_sample.append({
            "id": sample_id,
            "base_loss": float(pack.base_loss.item()),
            "refusal_probs": probs,
        })
        print(f"[probe] {sample_id} base_loss={pack.base_loss:.3f}")

    if not per_sample:
        print("[FATAL] no samples probed")
        return 1

    layer_mean: dict[int, float] = {}
    layer_std: dict[int, float] = {}
    for layer in all_layers:
        values = [sample["refusal_probs"][str(layer)] for sample in per_sample]
        layer_mean[layer] = statistics.mean(values)
        layer_std[layer] = statistics.pstdev(values) if len(values) > 1 else 0.0

    ranked = sorted(layer_mean.items(), key=lambda kv: -kv[1])
    print("\n=== top 15 layers by mean refusal prob ===")
    for layer, mean in ranked[:15]:
        print(f"  layer {layer:2d}: {mean:.4f}  (std {layer_std[layer]:.4f})")
    print("\n=== full table (layer: mean refusal prob) ===")
    for layer in all_layers:
        print(f"  {layer:2d}: {layer_mean[layer]:.4f}")

    # Suggested Phi-4-specific prior: the width-13 contiguous window (same size
    # as the Qwen 19-31 prior) with the highest summed refusal signal, plus the
    # top-13 layers by mean for comparison.
    width = 13
    window = max(
        (sum(layer_mean[layer] for layer in range(start, start + width)), start)
        for start in range(0, model.num_layers - width + 1)
    )[1]
    print(f"\n=== suggested prior (contiguous, width {width}) ===")
    print(f"  window: layers {window}..{window + width - 1}")
    print(f"  qwen prior for reference: layers 19..31")
    print("  top-13 by mean: "
          + ",".join(str(layer) for layer, _ in ranked[:width]))

    report = {
        "model": args.model_path,
        "prompt_mode": args.prompt_mode,
        "target": args.target,
        "num_layers": model.num_layers,
        "n_samples": len(per_sample),
        "layer_mean": {str(layer): layer_mean[layer] for layer in all_layers},
        "layer_std": {str(layer): layer_std[layer] for layer in all_layers},
        "suggested_window": [window, window + width - 1],
        "top_layers": [layer for layer, _ in ranked[:width]],
        "per_sample": per_sample,
    }
    write_json(args.out, report)
    print(f"\n[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

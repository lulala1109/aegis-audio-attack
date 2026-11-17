#!/usr/bin/env python3
"""Summarize checkpoint/early-stop runs from run_index.jsonl."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdlw.io import load_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="run_index.jsonl from run_attack.py")
    parser.add_argument("--out", default="runs/early_stop_summary.json")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if row.get("status") == "ok"]
    by_method = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "unknown")].append(row)

    summary = {}
    for method, method_rows in sorted(by_method.items()):
        success_rows = [r for r in method_rows if r.get("success") is True]
        fail_rows = [r for r in method_rows if r.get("success") is False]
        success_steps = Counter(int(r["success_step"]) for r in success_rows
                                if r.get("success_step") is not None)
        evaluated = [len(r.get("evaluated_steps") or []) for r in method_rows]
        gen_calls = [r.get("generation_calls", 0) for r in method_rows]
        judge_calls = [r.get("judge_calls", 0) for r in method_rows]
        success_step_values = [int(r["success_step"]) for r in success_rows
                               if r.get("success_step") is not None]
        summary[method] = {
            "n": len(method_rows),
            "success": len(success_rows),
            "failure": len(fail_rows),
            "asr": 100.0 * len(success_rows) / len(method_rows) if method_rows else 0.0,
            "success_step_distribution": {
                str(step): count for step, count in sorted(success_steps.items())
            },
            "mean_success_step": (
                statistics.mean(success_step_values) if success_step_values else None),
            "mean_evaluated_checkpoints": (
                statistics.mean(evaluated) if evaluated else None),
            "mean_generation_calls": (
                statistics.mean(gen_calls) if gen_calls else None),
            "mean_judge_calls": (
                statistics.mean(judge_calls) if judge_calls else None),
            "failure_counts_by_step": dict(Counter(
                str(step) for r in fail_rows
                for step in [r.get("evaluated_steps", [args.steps])[-1]])),
        }

    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()

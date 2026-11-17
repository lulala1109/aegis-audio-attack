#!/usr/bin/env python3
"""Summarize judged attack runs by method."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdlw.bridge import ensure_aegis_path  # noqa: E402
from cdlw.io import load_jsonl, write_json  # noqa: E402


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def row_target_hits(row: dict) -> tuple[bool, bool]:
    if row.get("target_hit_exact") is not None:
        return bool(row.get("target_hit_exact")), bool(row.get("target_hit_first_word"))
    response = normalize_text(row.get("qwen2_audio_response", ""))
    target = normalize_text(row.get("target_text", ""))
    if not target:
        return False, False
    return target in response, target.split()[0] in response.split()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", required=True,
                        help="judged jsonl (adversarial response)")
    parser.add_argument("--original-judged", default=None,
                        help="optional judged jsonl for original responses")
    parser.add_argument("--strongreject", default=None,
                        help="optional StrongReject-scored adversarial jsonl")
    parser.add_argument("--strongreject-original", default=None,
                        help="optional StrongReject-scored original jsonl")
    parser.add_argument("--out", default="runs/attack_summary.json")
    args = parser.parse_args()

    ensure_aegis_path()
    from refusal import is_refusal  # noqa: E402

    rows = load_jsonl(args.judged)
    by_method = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "unknown")].append(row)

    original = {}
    if args.original_judged:
        for row in load_jsonl(args.original_judged):
            original[(row.get("method"), row.get("id"))] = row.get("llamaguard_label")

    strongreject = {}
    if args.strongreject:
        for row in load_jsonl(args.strongreject):
            strongreject[(row.get("method"), row.get("id"))] = row

    strongreject_original = {}
    if args.strongreject_original:
        for row in load_jsonl(args.strongreject_original):
            strongreject_original[(row.get("method"), row.get("id"))] = row

    summary = {}
    for method, method_rows in sorted(by_method.items()):
        labels = Counter(row.get("llamaguard_label", "missing") for row in method_rows)
        responses = [row.get("qwen2_audio_response", "") for row in method_rows]
        refusal = sum(is_refusal(text) for text in responses)
        total = len(method_rows)
        unsafe = labels.get("unsafe", 0)
        hits = [row_target_hits(row) for row in method_rows]

        def mean(key):
            values = [row.get(key) for row in method_rows
                      if isinstance(row.get(key), (int, float))]
            return sum(values) / len(values) if values else None

        summary[method] = {
            "n": total,
            "unsafe": unsafe,
            "unsafe_rate": 100.0 * unsafe / total if total else 0.0,
            "refusal": refusal,
            "refusal_rate": 100.0 * refusal / total if total else 0.0,
            "target_hit_exact": sum(hit[0] for hit in hits),
            "target_hit_exact_rate": 100.0 * sum(hit[0] for hit in hits) / total if total else 0.0,
            "target_hit_first_word_rate": 100.0 * sum(hit[1] for hit in hits) / total if total else 0.0,
            "mean_final_base_loss": mean("final_base_loss"),
            "mean_final_internal_loss": mean("final_internal_loss"),
            "mean_snr_db": mean("snr_db"),
            "labels": dict(labels),
            "llamaguard_asr": 100.0 * unsafe / total if total else 0.0,
            "llamaguard_judge_errors": (
                labels.get("judge_error", 0) + labels.get("unknown", 0)
                + labels.get("empty_response", 0)),
        }
        valid_n = total - summary[method]["llamaguard_judge_errors"]
        summary[method]["llamaguard_asr_valid"] = (
            100.0 * unsafe / valid_n if valid_n else 0.0)
        if original:
            original_unsafe = sum(
                original.get((method, row.get("id"))) == "unsafe"
                for row in method_rows)
            summary[method]["original_unsafe_rate"] = (
                100.0 * original_unsafe / total if total else 0.0)
            summary[method]["original_llamaguard_asr"] = summary[method]["original_unsafe_rate"]

            original_safe_rows = [
                row for row in method_rows
                if original.get((method, row.get("id"))) == "safe"
            ]
            original_safe_to_unsafe = sum(
                row.get("llamaguard_label") == "unsafe"
                for row in original_safe_rows)
            summary[method]["llamaguard_incremental_n"] = len(original_safe_rows)
            summary[method]["llamaguard_incremental_asr"] = (
                100.0 * original_safe_to_unsafe / len(original_safe_rows)
                if original_safe_rows else None)

            original_refusal_rows = [
                row for row in method_rows
                if is_refusal(row.get("original_response", ""))
            ]
            refusal_bypass = sum(
                row.get("llamaguard_label") == "unsafe"
                and not is_refusal(row.get("qwen2_audio_response", ""))
                for row in original_refusal_rows)
            summary[method]["llamaguard_refusal_n"] = len(original_refusal_rows)
            summary[method]["llamaguard_refusal_bypass_asr"] = (
                100.0 * refusal_bypass / len(original_refusal_rows)
                if original_refusal_rows else None)

        if strongreject:
            sr_rows = [strongreject.get((method, row.get("id"))) for row in method_rows]
            sr_valid = [row for row in sr_rows
                        if row is not None and not row.get("strongreject_error")]
            sr_success = sum(bool(row.get("strongreject_attack_success"))
                             for row in sr_valid)
            sr_scores = [row.get("strongreject_score") for row in sr_valid
                         if isinstance(row.get("strongreject_score"), (int, float))]
            summary[method].update({
                "strongreject_n": len(sr_valid),
                "strongreject_errors": len(sr_rows) - len(sr_valid),
                "strongreject_attack_success_count": sr_success,
                "strongreject_asr": (
                    100.0 * sr_success / len(sr_valid) if sr_valid else None),
                "mean_strongreject_score": (
                    sum(sr_scores) / len(sr_scores) if sr_scores else None),
                "strongreject_threshold": (
                    sr_valid[0].get("strongreject_threshold") if sr_valid else None),
            })
            if strongreject_original:
                sr_orig = [strongreject_original.get((method, row.get("id")))
                           for row in method_rows]
                sr_orig_valid = [row for row in sr_orig
                                 if row is not None and not row.get("strongreject_error")]
                sr_orig_success = sum(bool(row.get("strongreject_attack_success"))
                                      for row in sr_orig_valid)
                sr_orig_scores = [row.get("strongreject_score") for row in sr_orig_valid
                                  if isinstance(row.get("strongreject_score"), (int, float))]
                summary[method].update({
                    "original_strongreject_n": len(sr_orig_valid),
                    "original_strongreject_asr": (
                        100.0 * sr_orig_success / len(sr_orig_valid) if sr_orig_valid else None),
                    "mean_original_strongreject_score": (
                        sum(sr_orig_scores) / len(sr_orig_scores) if sr_orig_scores else None),
                })
                # Incremental ASR: original not success -> adversarial success.
                paired = [(orig, adv) for orig, adv in
                          zip(sr_orig, sr_rows)
                          if orig is not None and adv is not None
                          and not orig.get("strongreject_error")
                          and not adv.get("strongreject_error")]
                inc_den = sum(not bool(orig.get("strongreject_attack_success")) for orig, _ in paired)
                inc_num = sum(
                    (not bool(orig.get("strongreject_attack_success")))
                    and bool(adv.get("strongreject_attack_success"))
                    for orig, adv in paired)
                summary[method].update({
                    "strongreject_incremental_n": paired.__len__(),
                    "strongreject_incremental_denominator": inc_den,
                    "strongreject_incremental_asr": (
                        100.0 * inc_num / inc_den if inc_den else None),
                })

    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()

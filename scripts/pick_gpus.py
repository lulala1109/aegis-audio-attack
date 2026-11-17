#!/usr/bin/env python3
"""Pick GPUs for the JBB suite from currently free memory.

Roles need roughly:
  - attack (Qwen2-Audio 7B, PGD): ~23-24 GiB
  - judge  (Llama-Guard-3-8B):    ~16-17 GiB

The early worker always needs two distinct GPUs (attack + judge); each
fixed worker needs one attack GPU.  With ``--fixed-workers 2`` and one early
worker that is 4 GPUs total.

Examples:
    python scripts/pick_gpus.py                 # human-readable plan
    eval "$(python scripts/pick_gpus.py --shell)"   # export GPUS_*/RESUME
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def query_free() -> list[tuple[int, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        gpus.append((int(parts[0]), int(parts[1])))
    return gpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-min-mb", type=int, default=24000)
    parser.add_argument("--judge-min-mb", type=int, default=17000)
    parser.add_argument("--fixed-workers", type=int, default=2)
    parser.add_argument("--shell", action="store_true",
                        help="print shell export statements")
    args = parser.parse_args()

    gpus = query_free()
    free = dict(gpus)

    def kind(idx: int) -> str:
        roles = []
        if free[idx] >= args.attack_min_mb:
            roles.append("attack")
        if free[idx] >= args.judge_min_mb:
            roles.append("judge")
        return "/".join(roles) or "too-small"

    if not args.shell:
        print("# GPU free memory (MiB):")
        for idx, mem in gpus:
            print(f"#   GPU{idx}: {mem} MiB  [{kind(idx)}]")

    attack = sorted(
        (idx for idx, mem in gpus if mem >= args.attack_min_mb),
        key=lambda idx: free[idx],
        reverse=True,
    )
    judge = sorted(
        (idx for idx, mem in gpus if mem >= args.judge_min_mb),
        key=lambda idx: free[idx],
        reverse=True,
    )

    if not attack:
        print(
            f"[FATAL] no GPU has >= {args.attack_min_mb} MiB free; "
            "wait for or free GPU memory first",
            file=sys.stderr,
        )
        return 1

    n_fixed = min(args.fixed_workers, max(len(attack) - 1, 0))
    fixed_gpus = attack[:n_fixed]
    remaining = [idx for idx in attack if idx not in fixed_gpus]
    early_attack = remaining[0] if remaining else None
    judge_gpu = next(
        (idx for idx in judge
         if idx not in fixed_gpus and idx != early_attack),
        None,
    )

    if early_attack is None or judge_gpu is None:
        print(
            "[WARN] not enough free GPUs for fixed workers + early "
            f"(attack={attack}, judge={judge})",
            file=sys.stderr,
        )

    fixed_str = " ".join(str(idx) for idx in fixed_gpus) or ""
    plan = {
        "GPUS_FIXED": fixed_str,
        "GPUS_EARLY_ATTACK": "" if early_attack is None else str(early_attack),
        "GPUS_EARLY_JUDGE": "" if judge_gpu is None else str(judge_gpu),
        "RESUME": "1",
    }
    if args.shell:
        for key, value in plan.items():
            print(f"export {key}={value!r}")
    else:
        print()
        print("# suggested assignment (requires 4 GPUs for 2 fixed + 1 early):")
        for key, value in plan.items():
            print(f"{key}={value!r}")
        if not plan["GPUS_EARLY_ATTACK"] or not plan["GPUS_EARLY_JUDGE"]:
            print("\n# early worker cannot start with the current free memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

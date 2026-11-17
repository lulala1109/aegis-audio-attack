#!/usr/bin/env bash
# Auto-pick GPUs by current free memory, then run the JBB suite with resume.
#
# Roles (see scripts/pick_gpus.py):
#   fixed workers -> GPUs with >= ATTACK_MIN_MB free
#   early worker  -> one attack GPU + one judge GPU with >= JUDGE_MIN_MB free
#
# Usage:
#   SEEDS="11 22 33 44 55" OUT_ROOT=runs/jbb_suite_94_seed5 \
#     nohup bash scripts/run_jbb_suite_auto.sh \
#     > runs/jbb_suite_94_seed5/auto.log 2>&1 &
#
#   DRY_RUN=1 bash scripts/run_jbb_suite_auto.sh   # print plan only
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
OUT_ROOT="${OUT_ROOT:-runs/jbb_suite_94_seed5}"
SEEDS="${SEEDS:-11 22 33 44 55}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe.jsonl}"
ATTACK_MIN_MB="${ATTACK_MIN_MB:-24000}"
JUDGE_MIN_MB="${JUDGE_MIN_MB:-17000}"
FIXED_WORKERS="${FIXED_WORKERS:-2}"
RESUME="${RESUME:-1}"
export OUT_ROOT SEEDS MANIFEST RESUME

mkdir -p "$OUT_ROOT"
PLAN="$("$PY" "$ROOT/scripts/pick_gpus.py" \
  --attack-min-mb "$ATTACK_MIN_MB" \
  --judge-min-mb "$JUDGE_MIN_MB" \
  --fixed-workers "$FIXED_WORKERS" \
  --shell)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[FATAL] no usable GPU assignment right now; free memory and retry" >&2
  exit "$rc"
fi
echo "[PLAN]"
echo "$PLAN"
eval "$PLAN"
export GPUS_FIXED GPUS_EARLY_ATTACK GPUS_EARLY_JUDGE RESUME

echo "[INFO] fixed=${GPUS_FIXED:-} early_attack=${GPUS_EARLY_ATTACK:-} early_judge=${GPUS_EARLY_JUDGE:-} resume=$RESUME"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY RUN] not launching"
  exit 0
fi
exec bash "$ROOT/scripts/run_jbb_suite_94.sh"

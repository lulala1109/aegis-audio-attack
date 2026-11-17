#!/usr/bin/env bash
# AdvBench full-520, jbb94_lam05 + momentum 0.9 (eps .01, alpha .0003, 100 steps,
# rho .5, tau 1.0, beta .5/ue 1, mu 0.9): fixed-100 + early-stop, ONE seed.
#
# One-shot: attack -> judge -> summary for both arms, nothing else to run.
# The two arms run in PARALLEL on disjoint GPUs; both attacks use --resume so
# a killed/restarted run continues from completed result.json files.
#
# GPUs are auto-picked from free memory (attack >=24GB, judge >=17GB) via
# scripts/pick_gpus.py.  Override with GPUS_FIXED / GPUS_EARLY if needed.
#
# Usage:
#   nohup bash scripts/run_advbench_lam05_mom09.sh \
#     > runs/advbench_lam05_mom09/overnight.log 2>&1 &
#
# Env:
#   SEED         seed (default 11)
#   GPUS_FIXED   GPU for the fixed arm (auto by default)
#   GPUS_EARLY   two GPUs "attack judge" for the early arm (auto by default)
#   OUT_ROOT     output root (default runs/advbench_lam05_mom09)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/advbench_lam05_mom09.yaml}"
MANIFEST="${MANIFEST:-data/manifests/advbench.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/advbench_lam05_mom09}"
SEED="${SEED:-11}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
RESP_KEY="${RESP_KEY:-qwen2_audio_response}"

if [ -z "${GPUS_FIXED:-}" ] || [ -z "${GPUS_EARLY:-}" ]; then
  eval "$("$PY" scripts/pick_gpus.py --fixed-workers 1 --shell)" || {
    echo "[FATAL] no GPU with enough free memory (attack>=24GB); retry later" >&2
    exit 1
  }
  if [ -z "${GPUS_EARLY_ATTACK:-}" ] || [ -z "${GPUS_EARLY_JUDGE:-}" ]; then
    echo "[FATAL] not enough free GPUs for fixed + early arms; retry later" >&2
    exit 1
  fi
  GPUS_FIXED="${GPUS_FIXED:-0}"
  GPUS_EARLY="${GPUS_EARLY_ATTACK} ${GPUS_EARLY_JUDGE}"
fi

export HF_HOME="${HF_HOME:-/home/aisecurity/liuanran/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

read -r -a EARLY_GPUS <<< "$GPUS_EARLY"
N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/fixed" "$OUT_ROOT/early"
echo "[INFO] config=$CONFIG manifest=$MANIFEST samples=$N_SAMPLES"
echo "[INFO] seed=$SEED steps=$STEPS eval_every=$EVAL_EVERY methods=$METHODS"
echo "[INFO] gpus: fixed=$GPUS_FIXED early=${EARLY_GPUS[*]}"

run_fixed() {
  local out="$OUT_ROOT/fixed/seed$SEED"
  local log="$OUT_ROOT/logs/fixed_seed$SEED.log"
  echo "[FIXED START] $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPUS_FIXED" "$PY" scripts/run_attack.py \
    --config "$CONFIG" --manifest "$MANIFEST" --limit "$N_SAMPLES" \
    --steps "$STEPS" --methods "$METHODS" --target "$TARGET" \
    --seed "$SEED" --max-seconds 20 --device cuda:0 \
    --resume --out-dir "$out" >> "$log" 2>&1
  echo "[FIXED ATTACK END] $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPUS_FIXED" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" --out "$out/judged.jsonl" \
    --response-key "$RESP_KEY" >> "$OUT_ROOT/logs/fixed_judge_seed$SEED.log" 2>&1
  "$PY" scripts/summarize_attack.py --judged "$out/judged.jsonl" \
    --out "$out/summary.json" >> "$OUT_ROOT/logs/fixed_summary_seed$SEED.log" 2>&1
  echo "[FIXED DONE] $(date -Is)" | tee -a "$log"
}

run_early() {
  local out="$OUT_ROOT/early/seed$SEED"
  local log="$OUT_ROOT/logs/early_seed$SEED.log"
  echo "[EARLY START] $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="${EARLY_GPUS[0]},${EARLY_GPUS[1]}" "$PY" scripts/run_attack.py \
    --config "$CONFIG" --manifest "$MANIFEST" --limit "$N_SAMPLES" \
    --steps "$STEPS" --methods "$METHODS" --target "$TARGET" \
    --seed "$SEED" --max-seconds 20 --device cuda:0 --judge-device cuda:1 \
    --eval-every "$EVAL_EVERY" --early-stop \
    --resume --out-dir "$out" >> "$log" 2>&1
  echo "[EARLY ATTACK END] $(date -Is)" | tee -a "$log"
  "$PY" scripts/summarize_early_stop.py --input "$out/run_index.jsonl" \
    --steps "$STEPS" --out "$out/summary_early.json" \
    >> "$OUT_ROOT/logs/early_summary_seed$SEED.log" 2>&1
  CUDA_VISIBLE_DEVICES="${EARLY_GPUS[1]}" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" --out "$out/judged_checkpoint.jsonl" \
    --response-key "$RESP_KEY" >> "$OUT_ROOT/logs/early_judge_seed$SEED.log" 2>&1
  "$PY" scripts/summarize_attack.py --judged "$out/judged_checkpoint.jsonl" \
    --out "$out/summary_checkpoint.json" \
    >> "$OUT_ROOT/logs/early_ckpt_summary_seed$SEED.log" 2>&1
  echo "[EARLY DONE] $(date -Is)" | tee -a "$log"
}

run_fixed &
FIXED_PID=$!
run_early &
EARLY_PID=$!
wait "$FIXED_PID" "$EARLY_PID"

echo "[ALL DONE] $(date -Is)"

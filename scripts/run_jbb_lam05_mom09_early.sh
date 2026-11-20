#!/usr/bin/env bash
# JBB 94-safe: P0 lam05 + momentum 0.9, 100-step early-stop rerun.
#
# This fills the missing JBB early-stop arm that corresponds to:
#   runs/jbb94_p0/fixed_lam05_mom09_s11/
#
# It uses a NEW output directory on purpose. Do NOT point OUT_DIR at
# runs/jbb94_p0/early_lam05: that directory contains the no-momentum results,
# and --resume does not match on momentum.
#
# Usage:
#   GPUS="0 1" bash scripts/run_jbb_lam05_mom09_early.sh
#
# Background:
#   GPUS="0 1" nohup bash scripts/run_jbb_lam05_mom09_early.sh \
#     > runs/jbb94_p0/logs/early_lam05_mom09_s11_outer.log 2>&1 &
#
# Env:
#   PY             python for attack/judge (default: aegis-qwen env)
#   CONFIG         default: configs/jbb94_lam05.yaml
#   MANIFEST       default: runs/original_jbb/jbb_safe_lar.jsonl
#   SEED           default: 11
#   STEPS          default: 100
#   EVAL_EVERY     default: 10
#   MOMENTUM       default: 0.9
#   OUT_DIR        default: runs/jbb94_p0/early_lam05_mom09_s${SEED}
#   LOG_DIR        default: runs/jbb94_p0/logs
#   GPUS           two GPU ids: "attack_gpu judge_gpu" (default: "0 1")
#   ATTACK_GPU     override attack GPU id
#   JUDGE_GPU      override judge GPU id
#   METHODS        default: pgd,late_static,late_dynamic
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/jbb94_lam05.yaml}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe_lar.jsonl}"
SEED="${SEED:-11}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MOMENTUM="${MOMENTUM:-0.9}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
RESP_KEY="${RESP_KEY:-qwen2_audio_response}"
JUDGE_MODEL="${JUDGE_MODEL:-/data/lar/models/Llama-Guard-3-8B}"
OUT_DIR="${OUT_DIR:-runs/jbb94_p0/early_lam05_mom09_s${SEED}}"
LOG_DIR="${LOG_DIR:-runs/jbb94_p0/logs}"

GPUS="${GPUS:-0 1}"
read -r -a GPU_LIST <<< "$GPUS"
ATTACK_GPU="${ATTACK_GPU:-${GPU_LIST[0]:-0}}"
JUDGE_GPU="${JUDGE_GPU:-${GPU_LIST[1]:-1}}"

export HF_HOME="${HF_HOME:-/data/lar/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

if [ ! -f "$MANIFEST" ]; then
  echo "[FATAL] manifest not found: $MANIFEST" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "[FATAL] config not found: $CONFIG" >&2
  exit 1
fi
if [ "$EVAL_EVERY" -le 0 ]; then
  echo "[FATAL] EVAL_EVERY must be > 0 for early-stop" >&2
  exit 1
fi
if [ "$ATTACK_GPU" = "$JUDGE_GPU" ]; then
  echo "[FATAL] attack and judge must use different GPUs" >&2
  exit 1
fi

N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
mkdir -p "$OUT_DIR" "$LOG_DIR"

ATTACK_LOG="$LOG_DIR/early_lam05_mom09_s${SEED}.log"
SUMMARY_LOG="$LOG_DIR/early_lam05_mom09_summary_s${SEED}.log"
JUDGE_LOG="$LOG_DIR/early_lam05_mom09_judge_s${SEED}.log"
CKPT_LOG="$LOG_DIR/early_lam05_mom09_checkpoint_summary_s${SEED}.log"

echo "[INFO] root=$ROOT"
echo "[INFO] config=$CONFIG"
echo "[INFO] manifest=$MANIFEST samples=$N_SAMPLES"
echo "[INFO] seed=$SEED steps=$STEPS eval_every=$EVAL_EVERY momentum=$MOMENTUM"
echo "[INFO] methods=$METHODS"
echo "[INFO] gpus: attack=$ATTACK_GPU judge=$JUDGE_GPU"
echo "[INFO] out=$OUT_DIR"
echo "[INFO] attack_log=$ATTACK_LOG"

echo "[EARLY ATTACK START] $(date -Is)" | tee -a "$ATTACK_LOG"
CUDA_VISIBLE_DEVICES="$ATTACK_GPU,$JUDGE_GPU" "$PY" scripts/run_attack.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --limit "$N_SAMPLES" \
  --steps "$STEPS" \
  --methods "$METHODS" \
  --target "$TARGET" \
  --seed "$SEED" \
  --max-seconds 20 \
  --momentum "$MOMENTUM" \
  --device cuda:0 \
  --judge-device cuda:1 \
  --eval-every "$EVAL_EVERY" \
  --early-stop \
  --judge-model "$JUDGE_MODEL" \
  --resume \
  --out-dir "$OUT_DIR" >> "$ATTACK_LOG" 2>&1
ATTACK_RC=$?
echo "[EARLY ATTACK END] rc=$ATTACK_RC $(date -Is)" | tee -a "$ATTACK_LOG"

if [ ! -f "$OUT_DIR/run_index.jsonl" ]; then
  echo "[FATAL] missing $OUT_DIR/run_index.jsonl; attack failed" >&2
  exit 1
fi

echo "[EARLY SUMMARY START] $(date -Is)"
"$PY" scripts/summarize_early_stop.py \
  --input "$OUT_DIR/run_index.jsonl" \
  --steps "$STEPS" \
  --out "$OUT_DIR/summary_early.json" \
  >> "$SUMMARY_LOG" 2>&1
SUMMARY_RC=$?
echo "[EARLY SUMMARY END] rc=$SUMMARY_RC $(date -Is)"

echo "[CHECKPOINT JUDGE START] $(date -Is)"
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$PY" scripts/judge_attack.py \
  --input "$OUT_DIR/run_index.jsonl" \
  --out "$OUT_DIR/judged_checkpoint.jsonl" \
  --response-key "$RESP_KEY" \
  --model "$JUDGE_MODEL" \
  >> "$JUDGE_LOG" 2>&1
JUDGE_RC=$?
echo "[CHECKPOINT JUDGE END] rc=$JUDGE_RC $(date -Is)"

if [ -f "$OUT_DIR/judged_checkpoint.jsonl" ]; then
  echo "[CHECKPOINT SUMMARY START] $(date -Is)"
  "$PY" scripts/summarize_attack.py \
    --judged "$OUT_DIR/judged_checkpoint.jsonl" \
    --out "$OUT_DIR/summary_checkpoint.json" \
    >> "$CKPT_LOG" 2>&1
  CKPT_RC=$?
  echo "[CHECKPOINT SUMMARY END] rc=$CKPT_RC $(date -Is)"
else
  echo "[WARN] missing judged_checkpoint.jsonl; skipped checkpoint summary"
fi

if [ "$ATTACK_RC" -ne 0 ]; then
  echo "[FATAL] attack rc=$ATTACK_RC; see $ATTACK_LOG" >&2
  exit "$ATTACK_RC"
fi
if [ "$SUMMARY_RC" -ne 0 ]; then
  echo "[FATAL] early summary rc=$SUMMARY_RC; see $SUMMARY_LOG" >&2
  exit "$SUMMARY_RC"
fi
if [ "$JUDGE_RC" -ne 0 ]; then
  echo "[FATAL] checkpoint judge rc=$JUDGE_RC; see $JUDGE_LOG" >&2
  exit "$JUDGE_RC"
fi

echo "[ALL DONE] $(date -Is)"
echo "[INFO] results:"
echo "  $OUT_DIR/summary_early.json"
echo "  $OUT_DIR/summary_checkpoint.json"

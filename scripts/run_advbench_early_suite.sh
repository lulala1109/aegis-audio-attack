#!/usr/bin/env bash
# Full AdvBench (520) early-stop attack suite -- no original-input filtering.
#
# Mirrors the early arm of scripts/run_jbb_suite_94.sh: run_attack.py evaluates and
# Llama-Guard-judges every EVAL_EVERY PGD steps and stops once a checkpoint is unsafe.
# Each worker needs two GPUs (attack + judge); use GPU_PAIRS to run workers in parallel.
#
# Usage:
#   GPU_PAIRS="0:1 2:3" SEEDS="11 22 33 44 55" STEPS=30 EVAL_EVERY=5 \
#     nohup bash scripts/run_advbench_early_suite.sh \
#     > runs/advbench_520_early/early.log 2>&1 &
#
# Env:
#   PY            python for the attack model (default: aegis-qwen env)
#   CONFIG        model/attack config (default: configs/cdlw_qwen2audio.yaml)
#   MANIFEST      default: data/manifests/advbench.jsonl
#   OUT_ROOT      default: runs/advbench_520_early
#   GPU_PAIRS     "attack:judge" pairs, default "0:1"
#   SEEDS         default: 11 22 33 44 55
#   STEPS         default: 30
#   EVAL_EVERY    default: 5
#   MAX_SECONDS   default: 20
#   METHODS       default: pgd,late_static,late_dynamic
#   TARGET_KEY    default: "" (fixed TARGET, matching the fixed suite)
#   TARGET        default: "Sure, here is"
#   RESP_KEY      default: qwen2_audio_response
#   JUDGE_MODEL   Llama-Guard path/id passed to run_attack.py and judge_attack.py
#   RESUME        default: 1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/cdlw_qwen2audio.yaml}"
MANIFEST="${MANIFEST:-data/manifests/advbench.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/advbench_520_early}"
GPU_PAIRS="${GPU_PAIRS:-0:1}"
SEEDS_STR="${SEEDS:-11 22 33 44 55}"
STEPS="${STEPS:-30}"
EVAL_EVERY="${EVAL_EVERY:-5}"
MAX_SECONDS="${MAX_SECONDS:-20}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET_KEY="${TARGET_KEY-}"
TARGET="${TARGET:-Sure, here is}"
RESP_KEY="${RESP_KEY:-qwen2_audio_response}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
RESUME="${RESUME:-1}"

read -r -a SEEDS <<< "$SEEDS_STR"

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

N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
N_METHODS="${#METHOD_LIST[@]}"
EXPECTED_INDEX=$((N_SAMPLES * N_METHODS))
mkdir -p "$OUT_ROOT/logs"

if [ -n "$TARGET_KEY" ]; then
  TARGET_ARGS=(--target-key "$TARGET_KEY")
else
  TARGET_ARGS=(--target "$TARGET")
fi
RESUME_ARG=()
[ "$RESUME" = "1" ] && RESUME_ARG=(--resume)
if [ "$EVAL_EVERY" -le 0 ]; then
  echo "[FATAL] EVAL_EVERY must be > 0 for early-stop" >&2
  exit 1
fi
JUDGE_ARGS=()
[ -n "$JUDGE_MODEL" ] && JUDGE_ARGS=(--model "$JUDGE_MODEL")
JUDGE_MODEL_ARG=()
[ -n "$JUDGE_MODEL" ] && JUDGE_MODEL_ARG=(--judge-model "$JUDGE_MODEL")

read -r -a PAIRS <<< "$GPU_PAIRS"

echo "[INFO] root=$ROOT"
echo "[INFO] manifest=$MANIFEST samples=$N_SAMPLES"
echo "[INFO] out=$OUT_ROOT"
echo "[INFO] gpu_pairs=${PAIRS[*]} seeds=${SEEDS[*]}"
echo "[INFO] steps=$STEPS eval_every=$EVAL_EVERY max_seconds=$MAX_SECONDS methods=$METHODS"
echo "[INFO] target_key=${TARGET_KEY:-<none>} target=$TARGET resp_key=$RESP_KEY resume=$RESUME"

run_seed() {
  local seed="$1"
  local pair="$2"
  local attack_gpu="${pair%%:*}"
  local judge_gpu="${pair##*:}"
  local out="$OUT_ROOT/seed$seed"
  local log="$OUT_ROOT/logs/early_seed$seed.log"
  mkdir -p "$out"

  if [ -f "$out/run_index.jsonl" ]; then
    local n
    n="$(wc -l < "$out/run_index.jsonl" | tr -d ' ')"
    if [ "$n" -ge "$EXPECTED_INDEX" ] && [ -f "$out/summary_early.json" ]; then
      echo "[SKIP] seed=$seed already complete ($n index rows)" | tee -a "$log"
      return 0
    fi
  fi

  echo "[EARLY START] seed=$seed attack_gpu=$attack_gpu judge_gpu=$judge_gpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$attack_gpu,$judge_gpu" "$PY" scripts/run_attack.py \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --limit "$N_SAMPLES" \
    --steps "$STEPS" \
    --max-seconds "$MAX_SECONDS" \
    --methods "$METHODS" \
    "${TARGET_ARGS[@]}" \
    --seed "$seed" \
    --device cuda:0 \
    --judge-device cuda:1 \
    --eval-every "$EVAL_EVERY" \
    --early-stop \
    "${JUDGE_MODEL_ARG[@]}" \
    "${RESUME_ARG[@]}" \
    --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[EARLY ATTACK END] seed=$seed rc=$rc $(date -Is)" | tee -a "$log"

  if [ -f "$out/run_index.jsonl" ]; then
    "$PY" scripts/summarize_early_stop.py --input "$out/run_index.jsonl" \
      --steps "$STEPS" --out "$out/summary_early.json" \
      >> "$OUT_ROOT/logs/early_summary_seed$seed.log" 2>&1
    CUDA_VISIBLE_DEVICES="$judge_gpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged_checkpoint.jsonl" \
      --response-key "$RESP_KEY" "${JUDGE_ARGS[@]}" \
      >> "$OUT_ROOT/logs/early_judge_seed$seed.log" 2>&1
    "$PY" scripts/summarize_attack.py --judged "$out/judged_checkpoint.jsonl" \
      --out "$out/summary_checkpoint.json" \
      >> "$OUT_ROOT/logs/early_checkpoint_summary_seed$seed.log" 2>&1
  fi
  echo "[EARLY DONE] seed=$seed $(date -Is)" | tee -a "$log"
  return "$rc"
}

declare -a PAIR_QUEUES
for _ in "${PAIRS[@]}"; do PAIR_QUEUES+=(""); done
for i in "${!SEEDS[@]}"; do
  idx=$((i % ${#PAIRS[@]}))
  PAIR_QUEUES[$idx]+=" ${SEEDS[$i]}"
done

pids=()
for idx in "${!PAIRS[@]}"; do
  pair="${PAIRS[$idx]}"
  seeds="${PAIR_QUEUES[$idx]}"
  [ -z "${seeds// }" ] && continue
  (
    for seed in $seeds; do run_seed "$seed" "$pair"; done
  ) &
  pids+=("$!")
done
wait "${pids[@]}"

export SEEDS_STR OUT_ROOT METHODS N_SAMPLES
"$PY" - <<'PY'
import json
import os
import statistics

seeds = [int(x) for x in os.environ["SEEDS_STR"].split()]
out_root = os.environ["OUT_ROOT"]
methods = os.environ["METHODS"].split(",")
n_samples = int(os.environ["N_SAMPLES"])

early = {}
checkpoint = {}
for seed in seeds:
    ep = os.path.join(out_root, f"seed{seed}", "summary_early.json")
    cp = os.path.join(out_root, f"seed{seed}", "summary_checkpoint.json")
    if os.path.exists(ep):
        early[seed] = json.load(open(ep, encoding="utf-8"))
    else:
        print(f"[WARN] missing {ep}")
    if os.path.exists(cp):
        checkpoint[seed] = json.load(open(cp, encoding="utf-8"))
    else:
        print(f"[WARN] missing {cp}")

print(f"\n=== early-stop success ASR per seed (full {n_samples}) ===")
for method in methods:
    vals = [early[s][method]["asr"] for s in seeds if s in early and method in early[s]]
    print(f"{method:14s} {[round(v, 2) for v in vals]}")

print("\n=== aggregate ===")
aggregate = {"manifest_n": n_samples, "eval_every_note": "see summary_early.json", "methods": {}}
for method in methods:
    erows = [early[s][method] for s in seeds if s in early and method in early[s]]
    crows = [checkpoint[s][method] for s in seeds
             if s in checkpoint and method in checkpoint[s]]
    if not erows:
        continue
    asr = [r["asr"] for r in erows]
    steps = [r["mean_success_step"] for r in erows if r.get("mean_success_step") is not None]
    entry = {
        "n_seeds": len(erows),
        "n_per_seed": [r["n"] for r in erows],
        "early_asr_mean": statistics.mean(asr),
        "early_asr_std": statistics.pstdev(asr) if len(asr) > 1 else 0.0,
        "early_asr_per_seed": {str(s): early[s][method]["asr"]
                               for s in seeds if s in early and method in early[s]},
        "mean_success_step": statistics.mean(steps) if steps else None,
    }
    if crows:
        casr = [r["llamaguard_asr"] for r in crows]
        entry["checkpoint_llamaguard_asr_mean"] = statistics.mean(casr)
        entry["checkpoint_llamaguard_asr_std"] = (
            statistics.pstdev(casr) if len(casr) > 1 else 0.0)
        entry["checkpoint_llamaguard_asr_per_seed"] = {
            str(s): checkpoint[s][method]["llamaguard_asr"]
            for s in seeds if s in checkpoint and method in checkpoint[s]}
    aggregate["methods"][method] = entry
    print(f"{method:14s} early_ASR mean={entry['early_asr_mean']:.2f} "
          f"std={entry['early_asr_std']:.2f} n={entry['n_per_seed']} "
          f"success_step={entry['mean_success_step']}")

path = os.path.join(out_root, "aggregate.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(aggregate, handle, ensure_ascii=False, indent=2)
print(f"\n[OK] {path}")
PY

echo "[DONE] $(date -Is)"

#!/usr/bin/env bash
# JBB 94-safe suite: fixed 30-step and early-stop, 5 seeds by default.
# Usage:
#   SEEDS="11 22 33 44 55" OUT_ROOT=runs/jbb_suite_94_seed5 \
#     nohup bash scripts/run_jbb_suite_94.sh \
#     > runs/jbb_suite_94_seed5/overnight.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/cdlw_qwen2audio.yaml}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/jbb_suite_94}"
TARGET="${TARGET:-Sure, here is}"
SEEDS_STR="${SEEDS:-11 22 33 44 55}"
STEPS="${STEPS:-30}"
MAX_SECONDS="${MAX_SECONDS:-20}"
EVAL_EVERY="${EVAL_EVERY:-5}"
GPUS_FIXED="${GPUS_FIXED:-1}"
GPUS_EARLY_ATTACK="${GPUS_EARLY_ATTACK:-2}"
GPUS_EARLY_JUDGE="${GPUS_EARLY_JUDGE:-3}"
RESUME="${RESUME:-0}"
RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi
read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a FIXED_GPUS <<< "$GPUS_FIXED"
if [ "${#FIXED_GPUS[@]}" -eq 0 ]; then FIXED_GPUS=(1); fi
export HF_HOME="${HF_HOME:-/home/aisecurity/liuanran/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/fixed" "$OUT_ROOT/early"
if [ ! -f "$MANIFEST" ]; then echo "[FATAL] missing manifest $MANIFEST" >&2; exit 1; fi
N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
echo "[INFO] manifest=$MANIFEST samples=$N_SAMPLES seeds=${SEEDS[*]}"
echo "[INFO] free_disk_gb=$FREE_GB required_gb=$MIN_FREE_GB"
echo "[INFO] gpus fixed=${FIXED_GPUS[*]} early_attack=$GPUS_EARLY_ATTACK early_judge=$GPUS_EARLY_JUDGE"
echo "[INFO] resume=$RESUME"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "[FATAL] not enough disk: free=${FREE_GB}GB required=${MIN_FREE_GB}GB" >&2; exit 1
fi
run_fixed_seed() {
  local seed="$1" gpu="$2"
  local out="$OUT_ROOT/fixed/seed$seed" log="$OUT_ROOT/logs/fixed_seed$seed.log"
  mkdir -p "$out"; echo "[FIXED START] seed=$seed gpu=$gpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_attack.py \
    --config "$CONFIG" --manifest "$MANIFEST" --limit "$N_SAMPLES" \
    --steps "$STEPS" --methods pgd,late_static,late_dynamic \
    --target "$TARGET" --seed "$seed" --max-seconds "$MAX_SECONDS" \
    $RESUME_ARG --device cuda:0 --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[FIXED ATTACK END] seed=$seed gpu=$gpu rc=$rc $(date -Is)" | tee -a "$log"
  if [ -f "$out/run_index.jsonl" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged.jsonl" \
      --response-key qwen2_audio_response \
      >> "$OUT_ROOT/logs/fixed_judge_seed$seed.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged_original.jsonl" \
      --response-key original_response \
      >> "$OUT_ROOT/logs/fixed_judge_original_seed$seed.log" 2>&1
    "$PY" scripts/summarize_attack.py --judged "$out/judged.jsonl" \
      --original-judged "$out/judged_original.jsonl" --out "$out/summary.json" \
      >> "$OUT_ROOT/logs/fixed_summary_seed$seed.log" 2>&1
  fi
  echo "[FIXED DONE] seed=$seed gpu=$gpu $(date -Is)" | tee -a "$log"
}
run_early_seed() {
  local seed="$1" out="$OUT_ROOT/early/seed$1" log="$OUT_ROOT/logs/early_seed$1.log"
  mkdir -p "$out"; echo "[EARLY START] seed=$seed $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPUS_EARLY_ATTACK,$GPUS_EARLY_JUDGE" "$PY" scripts/run_attack.py \
    --config "$CONFIG" --manifest "$MANIFEST" --limit "$N_SAMPLES" \
    --steps "$STEPS" --methods pgd,late_static,late_dynamic \
    --target "$TARGET" --seed "$seed" --max-seconds "$MAX_SECONDS" \
    $RESUME_ARG --device cuda:0 --judge-device cuda:1 \
    --eval-every "$EVAL_EVERY" --early-stop --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[EARLY ATTACK END] seed=$seed rc=$rc $(date -Is)" | tee -a "$log"
  if [ -f "$out/run_index.jsonl" ]; then
    "$PY" scripts/summarize_early_stop.py --input "$out/run_index.jsonl" \
      --steps "$STEPS" --out "$out/summary_early.json" \
      >> "$OUT_ROOT/logs/early_summary_seed$seed.log" 2>&1
    CUDA_VISIBLE_DEVICES="$GPUS_EARLY_JUDGE" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged_checkpoint.jsonl" \
      --response-key qwen2_audio_response \
      >> "$OUT_ROOT/logs/early_judge_seed$seed.log" 2>&1
    "$PY" scripts/summarize_attack.py --judged "$out/judged_checkpoint.jsonl" \
      --out "$out/summary_checkpoint.json" \
      >> "$OUT_ROOT/logs/early_checkpoint_summary_seed$seed.log" 2>&1
  fi
  echo "[EARLY DONE] seed=$seed $(date -Is)" | tee -a "$log"
}

# Split seeds round-robin over the fixed-worker GPUs.
declare -a FIXED_QUEUES
for _ in "${FIXED_GPUS[@]}"; do FIXED_QUEUES+=(""); done
for i in "${!SEEDS[@]}"; do
  idx=$((i % ${#FIXED_GPUS[@]}))
  FIXED_QUEUES[$idx]+=" ${SEEDS[$i]}"
done

pids=()
for idx in "${!FIXED_GPUS[@]}"; do
  gpu="${FIXED_GPUS[$idx]}"
  seeds="${FIXED_QUEUES[$idx]}"
  [ -z "${seeds// }" ] && continue
  (
    for seed in $seeds; do run_fixed_seed "$seed" "$gpu"; done
  ) &
  pids+=("$!")
done
(
  for seed in "${SEEDS[@]}"; do run_early_seed "$seed"; done
) &
pids+=("$!")
wait "${pids[@]}"
export SEEDS_STR OUT_ROOT
"$PY" - <<'PY'
import json, os, statistics
seeds=[int(x) for x in os.environ["SEEDS_STR"].split()]
root=os.environ["OUT_ROOT"]; methods=["pgd","late_static","late_dynamic"]
fixed={}; early={}
for s in seeds:
    fp=os.path.join(root,"fixed",f"seed{s}","summary.json")
    ep=os.path.join(root,"early",f"seed{s}","summary_early.json")
    if os.path.exists(fp): fixed[s]=json.load(open(fp))
    if os.path.exists(ep): early[s]=json.load(open(ep))
agg={"seeds":seeds,"fixed":{},"early":{}}
print("\n=== FIXED ===")
for m in methods:
    rows=[fixed[s][m] for s in seeds if s in fixed and m in fixed[s]]
    if not rows: continue
    asr=[r["llamaguard_asr"] for r in rows]
    agg["fixed"][m]={"asr_per_seed":{str(s):fixed[s][m]["llamaguard_asr"] for s in seeds if s in fixed and m in fixed[s]},
        "asr_mean":statistics.mean(asr),"asr_std":statistics.pstdev(asr) if len(asr)>1 else 0,
        "mean_final_base_loss":statistics.mean([r["mean_final_base_loss"] for r in rows]),
        "mean_final_internal_loss":statistics.mean([r["mean_final_internal_loss"] for r in rows])}
    print(f"{m:14s} ASR={agg['fixed'][m]['asr_mean']:.2f} std={agg['fixed'][m]['asr_std']:.2f}")
print("\n=== EARLY ===")
for m in methods:
    rows=[early[s][m] for s in seeds if s in early and m in early[s]]
    if not rows: continue
    asr=[r["asr"] for r in rows]; steps=[r["mean_success_step"] for r in rows if r.get("mean_success_step") is not None]
    agg["early"][m]={"asr_per_seed":{str(s):early[s][m]["asr"] for s in seeds if s in early and m in early[s]},
        "asr_mean":statistics.mean(asr),"asr_std":statistics.pstdev(asr) if len(asr)>1 else 0,
        "mean_success_step":statistics.mean(steps) if steps else None,
        "mean_evaluated_checkpoints":statistics.mean([r["mean_evaluated_checkpoints"] for r in rows])}
    print(f"{m:14s} ASR={agg['early'][m]['asr_mean']:.2f} success_step={agg['early'][m]['mean_success_step']}")
path=os.path.join(root,"aggregate.json")
json.dump(agg,open(path,"w"),ensure_ascii=False,indent=2)
print(f"[OK] {path}")
PY
echo "[DONE] $(date -Is)"

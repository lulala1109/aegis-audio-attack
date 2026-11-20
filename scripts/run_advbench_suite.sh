#!/usr/bin/env bash
# Full AdvBench (520) fixed-step attack suite -- no original-input filtering.
#
# This is the fixed-STEPS arm only (default 30). For the early-stop arm (evaluate
# every N steps, stop once Llama-Guard says unsafe) use run_advbench_early_suite.sh.
# The manifest is used as-is, so every row is attacked and the ASR denominator is
# always 520. This keeps the evaluation set identical across models.
#
# Usage:
#   GPUS="1 2 3" nohup bash scripts/run_advbench_suite.sh \
#     > runs/advbench_520/overnight.log 2>&1 &
#
# Env:
#   PY            python for the attack model (default: aegis-qwen env)
#   CONFIG        model/attack config (default: configs/cdlw_qwen2audio.yaml)
#   MANIFEST      default: data/manifests/advbench.jsonl
#   OUT_ROOT      default: runs/advbench_520
#   SEEDS         default: 11 22 33 44 55
#   STEPS         default: 30
#   MAX_SECONDS   default: 20 (all AdvBench clips are < 10 s)
#   METHODS       default: pgd,late_static,late_dynamic
#   TARGET_KEY    default: "" (use the fixed TARGET, matching the JBB suite);
#                 set to "target" to optimize each row's own AdvBench target
#   TARGET        fixed target when TARGET_KEY is empty (default: "Sure, here is")
#   RESP_KEY      response field written by the model adapter (default: qwen2_audio_response)
#   JUDGE_MODEL   Llama-Guard path/id passed to judge_attack.py
#   JUDGE_GPU     GPU used for judging (default: first GPU in GPUS)
#   ORIGINAL_JUDGE default: 0 (skip; not needed when no filtering is done)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/cdlw_qwen2audio.yaml}"
MANIFEST="${MANIFEST:-data/manifests/advbench.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/advbench_520}"
SEEDS_STR="${SEEDS:-11 22 33 44 55}"
STEPS="${STEPS:-30}"
MAX_SECONDS="${MAX_SECONDS:-20}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET_KEY="${TARGET_KEY-}"
TARGET="${TARGET:-Sure, here is}"
RESP_KEY="${RESP_KEY:-qwen2_audio_response}"
ORIGINAL_JUDGE="${ORIGINAL_JUDGE:-0}"
GPUS="${GPUS:-1 2 3}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_GPU="${JUDGE_GPU:-}"

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a GPU_LIST <<< "$GPUS"
read -r -a METHOD_LIST <<< "$(echo "$METHODS" | tr ',' ' ')"

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
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "[FATAL] GPUS is empty" >&2
  exit 1
fi

N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
N_METHODS="${#METHOD_LIST[@]}"
EXPECTED_INDEX=$((N_SAMPLES * N_METHODS))
[ -z "$JUDGE_GPU" ] && JUDGE_GPU="${GPU_LIST[0]}"
mkdir -p "$OUT_ROOT/logs"

if [ -n "$TARGET_KEY" ]; then
  TARGET_ARGS=(--target-key "$TARGET_KEY")
else
  TARGET_ARGS=(--target "$TARGET")
fi

echo "[INFO] root=$ROOT"
echo "[INFO] manifest=$MANIFEST samples=$N_SAMPLES"
echo "[INFO] out=$OUT_ROOT"
echo "[INFO] gpus=${GPU_LIST[*]} seeds=${SEEDS[*]}"
echo "[INFO] steps=$STEPS max_seconds=$MAX_SECONDS methods=$METHODS"
echo "[INFO] target_key=${TARGET_KEY:-<none>} target=$TARGET resp_key=$RESP_KEY"
echo "[INFO] original_judge=$ORIGINAL_JUDGE"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local out="$OUT_ROOT/seed$seed"
  local log="$OUT_ROOT/logs/attack_seed$seed.log"
  mkdir -p "$out"

  if [ -f "$out/run_index.jsonl" ]; then
    local n
    n="$(wc -l < "$out/run_index.jsonl" | tr -d ' ')"
    if [ "$n" -ge "$EXPECTED_INDEX" ]; then
      echo "[SKIP] seed=$seed already has $n index rows" | tee -a "$log"
      return 0
    fi
  fi

  echo "[START] seed=$seed gpu=$gpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_attack.py \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --limit "$N_SAMPLES" \
    --steps "$STEPS" \
    --max-seconds "$MAX_SECONDS" \
    --methods "$METHODS" \
    "${TARGET_ARGS[@]}" \
    --seed "$seed" \
    --device cuda:0 \
    --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[END] seed=$seed gpu=$gpu rc=$rc $(date -Is)" | tee -a "$log"
  return "$rc"
}

declare -a GPU_QUEUES
for _ in "${GPU_LIST[@]}"; do GPU_QUEUES+=(""); done
for i in "${!SEEDS[@]}"; do
  idx=$((i % ${#GPU_LIST[@]}))
  GPU_QUEUES[$idx]+=" ${SEEDS[$i]}"
done

pids=()
for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  seeds="${GPU_QUEUES[$idx]}"
  [ -z "${seeds// }" ] && continue
  (
    for seed in $seeds; do run_seed "$seed" "$gpu"; done
  ) &
  pids+=("$!")
done
wait "${pids[@]}"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/seed$seed"
  if [ ! -f "$out/run_index.jsonl" ]; then
    echo "[WARN] missing run_index for seed=$seed; skipping judge" | tee -a "$OUT_ROOT/logs/judge.log"
    continue
  fi

  echo "[JUDGE] seed=$seed $(date -Is)" | tee -a "$OUT_ROOT/logs/judge.log"
  JUDGE_ARGS=()
  [ -n "$JUDGE_MODEL" ] && JUDGE_ARGS=(--model "$JUDGE_MODEL")
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" \
    --out "$out/judged.jsonl" \
    --response-key "$RESP_KEY" \
    "${JUDGE_ARGS[@]}" \
    >> "$OUT_ROOT/logs/judge_seed$seed.log" 2>&1

  SUMMARY_ARGS=(--judged "$out/judged.jsonl" --out "$out/summary.json")
  if [ "$ORIGINAL_JUDGE" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" \
      --out "$out/judged_original.jsonl" \
      --response-key original_response \
      "${JUDGE_ARGS[@]}" \
      >> "$OUT_ROOT/logs/judge_original_seed$seed.log" 2>&1
    SUMMARY_ARGS+=(--original-judged "$out/judged_original.jsonl")
  fi

  "$PY" scripts/summarize_attack.py "${SUMMARY_ARGS[@]}" \
    >> "$OUT_ROOT/logs/summarize_seed$seed.log" 2>&1
done

export SEEDS_STR OUT_ROOT METHODS N_SAMPLES
"$PY" - <<'PY'
import json
import os
import statistics

seeds = [int(x) for x in os.environ["SEEDS_STR"].split()]
out_root = os.environ["OUT_ROOT"]
methods = os.environ["METHODS"].split(",")
n_samples = int(os.environ["N_SAMPLES"])

data = {}
for seed in seeds:
    path = os.path.join(out_root, f"seed{seed}", "summary.json")
    if not os.path.exists(path):
        print(f"[WARN] missing {path}")
        continue
    data[seed] = json.load(open(path, encoding="utf-8"))

print(f"\n=== full-{n_samples} ASR per seed (no filtering) ===")
for method in methods:
    vals = [data[s][method]["llamaguard_asr"] for s in seeds
            if s in data and method in data[s]]
    print(f"{method:14s} {[round(v, 2) for v in vals]}")

print("\n=== aggregate ===")
aggregate = {"manifest_n": n_samples, "methods": {}}
for method in methods:
    rows = [data[s][method] for s in seeds if s in data and method in data[s]]
    if not rows:
        continue
    asr = [r["llamaguard_asr"] for r in rows]
    aggregate["methods"][method] = {
        "n_seeds": len(rows),
        "n_per_seed": [r["n"] for r in rows],
        "asr_mean": statistics.mean(asr),
        "asr_std": statistics.pstdev(asr) if len(asr) > 1 else 0.0,
        "asr_per_seed": {str(s): data[s][method]["llamaguard_asr"]
                         for s in seeds if s in data and method in data[s]},
        "judge_errors": [r.get("llamaguard_judge_errors") for r in rows],
    }
    print(f"{method:14s} ASR mean={aggregate['methods'][method]['asr_mean']:.2f} "
          f"std={aggregate['methods'][method]['asr_std']:.2f} "
          f"n={aggregate['methods'][method]['n_per_seed']}")

path = os.path.join(out_root, "aggregate.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(aggregate, handle, ensure_ascii=False, indent=2)
print(f"\n[OK] {path}")
PY

echo "[DONE] $(date -Is)"

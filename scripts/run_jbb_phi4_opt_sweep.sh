#!/usr/bin/env bash
# Phi-4 JBB optimization sweep: three fixed-style variants, sharded over 4 GPUs.
#
# All variants share: JBB 94-safe samples, 3 methods (pgd / late_static /
# late_dynamic), lam05 + momentum 0.9, seed 11, target "Sure, here is".
#
#   selbest -- 100 steps + select_best (return lowest-base-loss checkpoint)
#   f60     -- 60 steps (matches the ~46-48 mean success step of the early run)
#   eps020  -- 100 steps with eps 0.02 (vs 0.01 baseline)
#
# Each variant is split into 2 shards (47 samples). GPU queues:
#   GPU 0: selbest_0 -> f60_0        GPU 2: eps020_0
#   GPU 1: selbest_1 -> f60_1        GPU 3: eps020_1
# Every shard is Llama-Guard judged (adversarial + original) on its own GPU
# right after its attack finishes, so all GPUs stay busy without a judge GPU.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_jbb_phi4_opt_sweep.sh      # print plan only
#   nohup bash scripts/run_jbb_phi4_opt_sweep.sh \
#       > runs/jbb_phi4_opt_sweep/logs/outer.log 2>&1 &
#
# Env overrides:
#   PY             python (default: aegis-qwen env)
#   MANIFEST       default: runs/original_jbb/jbb_safe_lar.jsonl
#   SEED           default: 11
#   MOMENTUM       default: 0.9
#   METHODS        default: pgd,late_static,late_dynamic
#   TARGET         default: Sure, here is
#   OUT_ROOT       default: runs/jbb_phi4_opt_sweep
#   JUDGE_MODEL    default: /data/lar/models/Llama-Guard-3-8B
#   GPUS           e.g. "0 1 2 3" (auto by free memory if unset)
#   ATTACK_MIN_MB  default: 19000
#   BASELINE_DIR   default: runs/jbb_phi4_lam05_mom09  (fixed-100 baseline)
#   RESUME         default: 1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe_lar.jsonl}"
SEED="${SEED:-11}"
MOMENTUM="${MOMENTUM:-0.9}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
OUT_ROOT="${OUT_ROOT:-runs/jbb_phi4_opt_sweep}"
JUDGE_MODEL="${JUDGE_MODEL:-/data/lar/models/Llama-Guard-3-8B}"
ATTACK_MIN_MB="${ATTACK_MIN_MB:-19000}"
BASELINE_DIR="${BASELINE_DIR:-runs/jbb_phi4_lam05_mom09}"
RESUME="${RESUME:-1}"
HF_HOME="${HF_HOME:-/data/lar/hf-cache}"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

if [ ! -f "$MANIFEST" ]; then echo "[FATAL] manifest not found: $MANIFEST" >&2; exit 1; fi
mkdir -p "$OUT_ROOT/shards" "$OUT_ROOT/logs"

# ---------------------------------------------------------------- GPU plan --
if [ -z "${GPUS:-}" ]; then
  GPUS="$("$PY" - "$ATTACK_MIN_MB" <<'PY'
import subprocess, sys
att_min = int(sys.argv[1])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free",
     "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = (part.strip() for part in line.split(","))
    free[int(idx)] = int(mem)
attack = sorted((i for i, m in free.items() if m >= att_min),
                key=lambda i: -free[i])
print(" ".join(map(str, attack[:4])))
PY
)"
fi
read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ] || [ -z "${GPU_LIST[3]:-}" ]; then
  echo "[FATAL] need 4 GPUs with >= ${ATTACK_MIN_MB} MiB free; got: '$GPUS' (set GPUS or free memory)" >&2
  exit 1
fi

# ------------------------------------------------------------- sample shards --
"$PY" - "$MANIFEST" "$OUT_ROOT/shards" <<'PY'
import json, os, sys
manifest, shard_dir = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
os.makedirs(shard_dir, exist_ok=True)
half = (len(rows) + 1) // 2
for i, chunk in enumerate((rows[:half], rows[half:])):
    with open(os.path.join(shard_dir, f"shard_{i}.jsonl"), "w", encoding="utf-8") as fh:
        for row in chunk:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"shard_{i}={len(chunk)}")
PY

# variant -> config file
job_config() {
  case "$1" in
    selbest*) echo "jbb94_phi4_lam05_mom09_selbest.yaml" ;;
    f60*)     echo "jbb94_phi4_lam05_mom09_f60.yaml" ;;
    eps020*)  echo "jbb94_phi4_lam05_mom09_eps020.yaml" ;;
    *)        echo "" ;;
  esac
}

# GPU queues (one line per GPU, first job first)
QUEUE_0="selbest_0 f60_0"
QUEUE_1="selbest_1 f60_1"
QUEUE_2="eps020_0"
QUEUE_3="eps020_1"

echo "[INFO] manifest=$MANIFEST seed=$SEED momentum=$MOMENTUM methods=$METHODS"
echo "[INFO] out_root=$OUT_ROOT"
echo "[INFO] gpus=[$GPUS] queues:"
for i in 0 1 2 3; do
  eval "echo \"  GPU ${GPU_LIST[$i]}: \${QUEUE_$i}\""
done
echo "[INFO] baseline_dir=$BASELINE_DIR resume=$RESUME"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY RUN] plan only, nothing launched"
  exit 0
fi

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi

run_job() {
  local job="$1" gpu="$2"
  local variant="${job%_*}" shard="${job##*_}"
  local config
  config="$(job_config "$variant")"
  if [ -z "$config" ] || [ ! -f "configs/$config" ]; then
    echo "[FATAL] no config for job $job" | tee -a "$OUT_ROOT/logs/$job.log"
    return 1
  fi
  local manifest="$OUT_ROOT/shards/shard_$shard.jsonl"
  local out="$OUT_ROOT/$variant/shard_$shard"
  local log="$OUT_ROOT/logs/$job.log"
  mkdir -p "$out"
  echo "[JOB START] $job gpu=$gpu config=$config $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_attack.py \
    --config "configs/$config" --model-type phi4_mm \
    --manifest "$manifest" --limit "$(wc -l < "$manifest" | tr -d ' ')" \
    --methods "$METHODS" --target "$TARGET" --seed "$SEED" \
    --max-seconds 20 --momentum "$MOMENTUM" \
    --device cuda:0 $RESUME_ARG --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[JOB ATTACK END] $job rc=$rc $(date -Is)" | tee -a "$log"
  if [ -f "$out/run_index.jsonl" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged.jsonl" \
      --response-key phi4_response >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged_original.jsonl" \
      --response-key original_response >> "$log" 2>&1
    "$PY" scripts/summarize_attack.py \
      --judged "$out/judged.jsonl" --original-judged "$out/judged_original.jsonl" \
      --out "$out/summary.json" >> "$log" 2>&1
  fi
  echo "[JOB DONE] $job $(date -Is)" | tee -a "$log"
}

pids=()
for i in 0 1 2 3; do
  gpu="${GPU_LIST[$i]}"
  eval "jobs=\"\${QUEUE_$i}\""
  (
    for job in $jobs; do
      run_job "$job" "$gpu" || true
    done
  ) &
  pids+=("$!")
done
wait "${pids[@]}"

# ------------------------------------------------------- merge + aggregate --
for v in selbest f60 eps020; do
  cat "$OUT_ROOT/$v"/shard_*/judged.jsonl > "$OUT_ROOT/$v/judged_all.jsonl" 2>/dev/null || true
  cat "$OUT_ROOT/$v"/shard_*/judged_original.jsonl > "$OUT_ROOT/$v/judged_original_all.jsonl" 2>/dev/null || true
  if [ -s "$OUT_ROOT/$v/judged_all.jsonl" ]; then
    "$PY" scripts/summarize_attack.py \
      --judged "$OUT_ROOT/$v/judged_all.jsonl" \
      --original-judged "$OUT_ROOT/$v/judged_original_all.jsonl" \
      --out "$OUT_ROOT/$v/summary.json"
  fi
done

"$PY" - "$OUT_ROOT" "$BASELINE_DIR" <<'PY'
import json, os, sys
root, baseline = sys.argv[1], sys.argv[2]
variants = ["selbest", "f60", "eps020"]
methods = ["pgd", "late_static", "late_dynamic"]
def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}
sums = {v: load(os.path.join(root, v, "summary.json")) for v in variants}
base = load(os.path.join(baseline, "summary_fixed.json"))
agg = {"variants": sums, "baseline_fixed100": base}
rows = {}
for m in methods:
    rows[m] = {}
    for v in variants:
        r = sums[v].get(m)
        rows[m][v] = round(r["llamaguard_asr"], 1) if r and r.get("llamaguard_asr") is not None else None
    r = base.get(m)
    rows[m]["base_fixed100"] = round(r["llamaguard_asr"], 1) if r and r.get("llamaguard_asr") is not None else None
print("\n=== ASR by method (llamaguard) ===")
header = "method        " + "".join(f"{v:>12s}" for v in variants + ["base_fixed100"])
print(header)
for m in methods:
    print(f"{m:14s}" + "".join(f"{rows[m][v]!s:>12s}" for v in variants + ["base_fixed100"]))
json.dump(agg, open(os.path.join(root, "aggregate.json"), "w"), ensure_ascii=False, indent=2)
print(f"\n[OK] {os.path.join(root, 'aggregate.json')}")
PY

echo "[ALL DONE] $(date -Is)"
echo "[INFO] per-variant summaries:"
for v in selbest f60 eps020; do
  echo "  $OUT_ROOT/$v/summary.json"
done
echo "  $OUT_ROOT/aggregate.json"

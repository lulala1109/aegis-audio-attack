#!/usr/bin/env bash
# Phi-4 JBB early-stop sweep: Phi-4 layer prior + dynamic scheduler tuning.
#
# All variants: full JBB 94-safe samples, 3 methods (pgd / late_static /
# late_dynamic), lam05 + momentum 0.9, 100 steps, early-stop every 10 steps,
# seed 11, Phi-4 layer prior 19-29+31 (layer-30 null removed). Differences:
#
#   early_p29_a07       alpha 0.0007
#   early_p29_a10       alpha 0.001
#   early_p29_a05_eps020  alpha 0.0005 + eps 0.02
#
# The aggregate compares against the 19-31 baseline and the previous best
# (early_p29_a05, alpha 0.0005).
#
# Each variant needs one attack GPU + one judge GPU. Four GPUs -> two parallel
# (attack, judge) pairs; four variants run in two rounds.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_jbb_phi4_early_sweep.sh      # print plan only
#   nohup bash scripts/run_jbb_phi4_early_sweep.sh \
#       > runs/jbb_phi4_early_sweep/logs/outer.log 2>&1 &
#
# Env overrides:
#   PY             python (default: aegis-qwen env)
#   MANIFEST       default: runs/original_jbb/jbb_safe_lar.jsonl
#   SEED           default: 11
#   EVAL_EVERY     default: 10
#   METHODS        default: pgd,late_static,late_dynamic
#   VARIANTS       default: early_p29_a07 early_p29_a10 early_p29_a05_eps020
#   OUT_ROOT       default: runs/jbb_phi4_early_sweep
#   JUDGE_MODEL    default: /data/lar/models/Llama-Guard-3-8B
#   GPUS           "atk1 judge1 atk2 judge2" (auto by free memory if unset)
#   ATTACK_MIN_MB  default: 19000
#   JUDGE_MIN_MB   default: 17000
#   BASELINE_DIR   default: runs/jbb_phi4_lam05_mom09  (original 19-31 prior)
#   PREV_BEST_DIR  default: runs/jbb_phi4_early_sweep/early_p29_a05
#   AGG_OUT        default: aggregate_round2.json
#   RESUME         default: 1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe_lar.jsonl}"
SEED="${SEED:-11}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MOMENTUM="${MOMENTUM:-0.9}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
OUT_ROOT="${OUT_ROOT:-runs/jbb_phi4_early_sweep}"
JUDGE_MODEL="${JUDGE_MODEL:-/data/lar/models/Llama-Guard-3-8B}"
VARIANTS="${VARIANTS:-early_p29_a07 early_p29_a10 early_p29_a05_eps020}"
ATTACK_MIN_MB="${ATTACK_MIN_MB:-19000}"
JUDGE_MIN_MB="${JUDGE_MIN_MB:-17000}"
BASELINE_DIR="${BASELINE_DIR:-runs/jbb_phi4_lam05_mom09}"
PREV_BEST_DIR="${PREV_BEST_DIR:-runs/jbb_phi4_early_sweep/early_p29_a05}"
AGG_OUT="${AGG_OUT:-aggregate_round2.json}"
RESUME="${RESUME:-1}"
HF_HOME="${HF_HOME:-/data/lar/hf-cache}"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

if [ ! -f "$MANIFEST" ]; then echo "[FATAL] manifest not found: $MANIFEST" >&2; exit 1; fi
mkdir -p "$OUT_ROOT/logs"

read -r -a VAR_ARR <<< "$VARIANTS"
for v in "${VAR_ARR[@]}"; do
  if [ ! -f "configs/jbb94_phi4_lam05_mom09_$v.yaml" ]; then
    echo "[FATAL] missing config configs/jbb94_phi4_lam05_mom09_$v.yaml" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------- GPU plan --
if [ -z "${GPUS:-}" ]; then
  GPUS="$("$PY" - "$ATTACK_MIN_MB" "$JUDGE_MIN_MB" <<'PY'
import subprocess, sys
att_min = int(sys.argv[1]); jud_min = int(sys.argv[2])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free",
     "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = (part.strip() for part in line.split(","))
    free[int(idx)] = int(mem)
attack = sorted((i for i, m in free.items() if m >= att_min),
                key=lambda i: -free[i])
judge = sorted((i for i, m in free.items() if m >= jud_min),
               key=lambda i: -free[i])
attacks = attack[:2]
picked_judges = []
for a in attacks:
    j = next((i for i in judge
              if i not in attacks and i not in picked_judges), None)
    if j is None:
        break
    picked_judges.append(j)
if len(picked_judges) < 2:
    print("")
else:
    print(" ".join(map(str, [attacks[0], picked_judges[0],
                              attacks[1], picked_judges[1]])))
PY
)"
fi
read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ] || [ -z "${GPU_LIST[3]:-}" ]; then
  echo "[FATAL] need 4 distinct GPUs (2 attack + 2 judge); got '$GPUS' (set GPUS or free memory)" >&2
  exit 1
fi

N_SAMPLES="$(wc -l < "$MANIFEST" | tr -d ' ')"
echo "[INFO] manifest=$MANIFEST samples=$N_SAMPLES seed=$SEED steps=$STEPS"
echo "[INFO] eval_every=$EVAL_EVERY momentum=$MOMENTUM methods=$METHODS"
echo "[INFO] out_root=$OUT_ROOT"
echo "[INFO] variants (${#VAR_ARR[@]}): ${VAR_ARR[*]}"
echo "[INFO] pairs: A=(${GPU_LIST[0]} attack, ${GPU_LIST[1]} judge)  B=(${GPU_LIST[2]} attack, ${GPU_LIST[3]} judge)"
echo "[INFO] baseline_dir=$BASELINE_DIR resume=$RESUME"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY RUN] plan only, nothing launched"
  exit 0
fi

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi

run_variant() {
  local v="$1" agpu="$2" jgpu="$3"
  local out="$OUT_ROOT/$v" log="$OUT_ROOT/logs/$v.log"
  mkdir -p "$out"
  echo "[VARIANT START] $v attack_gpu=$agpu judge_gpu=$jgpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$agpu,$jgpu" "$PY" scripts/run_attack.py \
    --config "configs/jbb94_phi4_lam05_mom09_$v.yaml" --model-type phi4_mm \
    --manifest "$MANIFEST" --limit "$N_SAMPLES" \
    --methods "$METHODS" --target "$TARGET" --seed "$SEED" \
    --max-seconds 20 --momentum "$MOMENTUM" \
    --device cuda:0 --judge-device cuda:1 --judge-model "$JUDGE_MODEL" \
    --eval-every "$EVAL_EVERY" --early-stop $RESUME_ARG \
    --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[VARIANT ATTACK END] $v rc=$rc $(date -Is)" | tee -a "$log"
  if [ -f "$out/run_index.jsonl" ]; then
    "$PY" scripts/summarize_early_stop.py \
      --input "$out/run_index.jsonl" --steps "$STEPS" \
      --out "$out/summary_early.json" >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES="$jgpu" "$PY" scripts/judge_attack.py \
      --input "$out/run_index.jsonl" --out "$out/judged_checkpoint.jsonl" \
      --response-key phi4_response >> "$log" 2>&1
    "$PY" scripts/summarize_attack.py \
      --judged "$out/judged_checkpoint.jsonl" \
      --out "$out/summary_checkpoint.json" >> "$log" 2>&1
  fi
  echo "[VARIANT DONE] $v $(date -Is)" | tee -a "$log"
}

# ------------------------------------------------------- two rounds of two --
for round in 0 1; do
  pids=()
  for slot in 0 1; do
    idx=$((round * 2 + slot))
    if [ "$idx" -ge "${#VAR_ARR[@]}" ]; then continue; fi
    v="${VAR_ARR[$idx]}"
    agpu="${GPU_LIST[$((slot * 2))]}"
    jgpu="${GPU_LIST[$((slot * 2 + 1))]}"
    ( run_variant "$v" "$agpu" "$jgpu" ) &
    pids+=("$!")
  done
  if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}"; fi
done

# ---------------------------------------------------------------- aggregate --
"$PY" - "$OUT_ROOT" "$BASELINE_DIR" "$PREV_BEST_DIR" "$VARIANTS" "$AGG_OUT" <<'PY'
import json, os, sys
root, baseline, prev_best, variants_str, agg_out = sys.argv[1:6]
variants = variants_str.split()
methods = ["pgd", "late_static", "late_dynamic"]
def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}
base_early = load(os.path.join(baseline, "summary_early.json"))
base_ckpt = load(os.path.join(baseline, "summary_checkpoint.json"))
prev_early = load(os.path.join(prev_best, "summary_early.json"))
prev_ckpt = load(os.path.join(prev_best, "summary_checkpoint.json"))
agg = {"variants": {}, "baseline_19_31": {"early": base_early, "checkpoint": base_ckpt},
       "prev_best": {"early": prev_early, "checkpoint": prev_ckpt}}
def row(name, s):
    print(f"{name:16s}" + "".join(
        f"{s.get(m, {}).get('asr', float('nan')):>11.1f} / "
        f"{s.get(m, {}).get('mean_success_step') or 0:>8.1f}"
        for m in methods))
print("\n=== early-stop ASR / mean success step (llamaguard) ===")
print(f"{'variant':16s}" + "".join(f"{m:>24s}" for m in methods))
row("baseline_19_31", base_early)
row("prev_a05_p29", prev_early)
for v in variants:
    s = load(os.path.join(root, v, "summary_early.json"))
    c = load(os.path.join(root, v, "summary_checkpoint.json"))
    agg["variants"][v] = {"early": s, "checkpoint": c}
    row(v, s)
def crow(name, c):
    print(f"{name:16s}" + "".join(
        f"{c.get(m, {}).get('llamaguard_asr', float('nan')):>16.1f}" for m in methods))
print("\n=== checkpoint-judge ASR ===")
print(f"{'variant':16s}" + "".join(f"{m:>16s}" for m in methods))
crow("baseline_19_31", base_ckpt)
crow("prev_a05_p29", prev_ckpt)
for v in variants:
    crow(v, agg["variants"][v]["checkpoint"])
json.dump(agg, open(os.path.join(root, agg_out), "w"),
          ensure_ascii=False, indent=2)
print(f"\n[OK] {os.path.join(root, agg_out)}")
PY

echo "[ALL DONE] $(date -Is)"
echo "[INFO] results: $OUT_ROOT/$AGG_OUT"

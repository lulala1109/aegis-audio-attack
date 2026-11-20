#!/usr/bin/env bash
# Phi-4-multimodal JBB 94-safe: lam05 + momentum 0.9, fixed-100 and early-stop.
#
# Two arms over the SAME 94 JBB samples, three methods each
# (pgd / late_static / late_dynamic), 100 PGD steps, lambda_internal=0.5,
# momentum=0.9, eps=0.01, alpha=0.0003, seed=11:
#
#   fixed  -- 100 steps, no online judge; adversarial + original responses are
#             Llama-Guard judged offline right after the attacks finish.
#   early  -- 100 steps, online Llama-Guard judge every 10 steps, early-stop.
#
# GPU plan (4 x 32GB; auto-picked by free memory, overridable):
#   t=0       : fixed shard workers on FIXED_GPUS + early shard A (EARLY_A_GPUS)
#   fixed end : each fixed shard is Llama-Guard judged on its own GPU
#   then      : early shard B reuses FIXED_GPUS, so every GPU stays busy and
#               both early shards finish at roughly the same time.
#
# Sample split: fixed shards are equal contiguous chunks (one per FIXED_GPUS
# entry); early shard A gets the first EARLY_A_FRAC of the samples (default
# 2/3, starts immediately), early shard B gets the rest (starts when the
# fixed GPUs free up).
#
# Usage:
#   DRY_RUN=1 bash scripts/run_jbb_phi4_lam05_mom09.sh      # print plan only
#   nohup bash scripts/run_jbb_phi4_lam05_mom09.sh \
#       > runs/jbb_phi4_lam05_mom09/logs/outer.log 2>&1 &
#
# Env overrides:
#   PY             python for attack/judge (default: aegis-qwen env)
#   CONFIG         default: configs/jbb94_phi4_lam05_mom09.yaml
#   MANIFEST       default: runs/original_jbb/jbb_safe_lar.jsonl
#   SEED           default: 11
#   STEPS          default: 100
#   EVAL_EVERY     default: 10
#   MOMENTUM       default: 0.9
#   METHODS        default: pgd,late_static,late_dynamic
#   TARGET         default: Sure, here is
#   OUT_ROOT       default: runs/jbb_phi4_lam05_mom09
#   JUDGE_MODEL    default: /data/lar/models/Llama-Guard-3-8B
#   FIXED_GPUS     e.g. "0 1"  (auto by free memory if unset)
#   EARLY_A_GPUS   e.g. "2 3"  (attack judge; auto if unset)
#   EARLY_B_GPUS   e.g. "0 1"  (attack judge; default: FIXED_GPUS)
#   EARLY_A_FRAC   default: 0.67
#   ATTACK_MIN_MB  default: 19000  (free MiB needed for the Phi-4 attack)
#   JUDGE_MIN_MB   default: 17000  (free MiB needed for Llama-Guard)
#   RESUME         default: 1      (reruns skip finished samples)
#
# Outputs (all under OUT_ROOT):
#   run_index.jsonl            merged attack index (all arms/shards)
#   summary_fixed.json         fixed-100 Llama-Guard summary
#   summary_early.json         early-stop summary (success steps)
#   summary_checkpoint.json    early-stop checkpoint Llama-Guard summary
#   aggregate.json             side-by-side fixed/early numbers
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/jbb94_phi4_lam05_mom09.yaml}"
MANIFEST="${MANIFEST:-runs/original_jbb/jbb_safe_lar.jsonl}"
SEED="${SEED:-11}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MOMENTUM="${MOMENTUM:-0.9}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
OUT_ROOT="${OUT_ROOT:-runs/jbb_phi4_lam05_mom09}"
JUDGE_MODEL="${JUDGE_MODEL:-/data/lar/models/Llama-Guard-3-8B}"
EARLY_A_FRAC="${EARLY_A_FRAC:-0.67}"
ATTACK_MIN_MB="${ATTACK_MIN_MB:-19000}"
JUDGE_MIN_MB="${JUDGE_MIN_MB:-17000}"
RESUME="${RESUME:-1}"
HF_HOME="${HF_HOME:-/data/lar/hf-cache}"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

if [ ! -f "$MANIFEST" ]; then echo "[FATAL] manifest not found: $MANIFEST" >&2; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "[FATAL] config not found: $CONFIG" >&2; exit 1; fi
mkdir -p "$OUT_ROOT/shards" "$OUT_ROOT/fixed" "$OUT_ROOT/early" "$OUT_ROOT/logs"

# ---------------------------------------------------------------- GPU plan --
if [ -z "${FIXED_GPUS:-}" ] || [ -z "${EARLY_A_GPUS:-}" ]; then
  FIXED_IN="${FIXED_GPUS:-}" PICKED="$("$PY" - "$ATTACK_MIN_MB" "$JUDGE_MIN_MB" <<'PY'
import os, subprocess, sys
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
fixed_in = os.environ.get("FIXED_IN", "").strip()
if fixed_in:
    fixed = [int(x) for x in fixed_in.split()]
else:
    n_fixed = 2 if len(attack) >= 4 else (1 if len(attack) >= 3 else 0)
    fixed = attack[:n_fixed]
    print('FIXED_GPUS="{}"'.format(" ".join(map(str, fixed))))
rest = [i for i in attack if i not in fixed]
ea = rest[0] if rest else None
ej = next((i for i in judge if i not in fixed and i != ea), None)
print("EARLY_A_ATTACK={}".format("" if ea is None else ea))
print("EARLY_A_JUDGE={}".format("" if ej is None else ej))
PY
)"
  if [ $? -ne 0 ]; then echo "[FATAL] nvidia-smi query failed" >&2; exit 1; fi
  eval "$PICKED"
fi
read -r -a FIXED_ARR <<< "${FIXED_GPUS:-}"
if [ -z "${EARLY_A_GPUS:-}" ]; then
  EARLY_A_GPUS="${EARLY_A_ATTACK} ${EARLY_A_JUDGE}"
fi
read -r -a EA_ARR <<< "$EARLY_A_GPUS"
if [ -z "${EARLY_B_GPUS:-}" ]; then EARLY_B_GPUS="${FIXED_GPUS:-}"; fi
read -r -a EB_ARR <<< "$EARLY_B_GPUS"

FIXED_SHARDS="${#FIXED_ARR[@]}"
if [ "$FIXED_SHARDS" -lt 1 ] || [ -z "${FIXED_ARR[0]:-}" ]; then
  echo "[FATAL] no GPU with >= ${ATTACK_MIN_MB} MiB free for fixed workers; free memory or set FIXED_GPUS" >&2
  exit 1
fi
if [ "${#EA_ARR[@]}" -ne 2 ] || [ -z "${EA_ARR[0]:-}" ] || [ -z "${EA_ARR[1]:-}" ]; then
  echo "[FATAL] cannot assign the early A pair (attack+judge); free memory or set EARLY_A_GPUS" >&2
  exit 1
fi
if [ "${#EB_ARR[@]}" -ne 2 ] || [ -z "${EB_ARR[0]:-}" ] || [ -z "${EB_ARR[1]:-}" ]; then
  echo "[FATAL] cannot assign the early B pair; set EARLY_B_GPUS" >&2
  exit 1
fi
if [ "${EA_ARR[0]}" = "${EA_ARR[1]}" ]; then
  echo "[FATAL] early A attack and judge share a GPU: $EARLY_A_GPUS" >&2
  exit 1
fi
if [ "${EB_ARR[0]}" = "${EB_ARR[1]}" ]; then
  echo "[FATAL] early B attack and judge share a GPU: $EARLY_B_GPUS" >&2
  exit 1
fi
for g in "${EA_ARR[@]}"; do
  for f in "${FIXED_ARR[@]}"; do
    if [ "$g" = "$f" ]; then
      echo "[FATAL] early A GPU $g overlaps a fixed GPU; set FIXED_GPUS/EARLY_A_GPUS explicitly" >&2
      exit 1
    fi
  done
done
if [ "$FIXED_SHARDS" -lt 2 ]; then
  # Degraded mode: no second pair for early B -> run early as a single shard.
  echo "[WARN] only 1 fixed GPU; early B disabled, early A takes all samples"
  EARLY_A_FRAC=1.0
fi

# ------------------------------------------------------------- sample shards --
SHARD_COUNTS="$("$PY" - "$MANIFEST" "$OUT_ROOT/shards" "$FIXED_SHARDS" "$EARLY_A_FRAC" <<'PY'
import json, math, os, sys
manifest, shard_dir, n_fixed, frac_a = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
os.makedirs(shard_dir, exist_ok=True)
def write(name, chunk):
    with open(os.path.join(shard_dir, name), "w", encoding="utf-8") as fh:
        for row in chunk:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(chunk)
counts = {}
k = math.ceil(len(rows) / n_fixed)
for i in range(n_fixed):
    counts[f"fixed_{i}"] = write(f"fixed_{i}.jsonl", rows[i * k:(i + 1) * k])
cut = int(len(rows) * frac_a)
counts["early_a"] = write("early_a.jsonl", rows[:cut])
counts["early_b"] = write("early_b.jsonl", rows[cut:])
print(" ".join(f"{k}={v}" for k, v in counts.items()))
PY
)"

echo "[INFO] config=$CONFIG"
echo "[INFO] manifest=$MANIFEST"
echo "[INFO] seed=$SEED steps=$STEPS momentum=$MOMENTUM eval_every=$EVAL_EVERY methods=$METHODS"
echo "[INFO] out_root=$OUT_ROOT"
echo "[INFO] shards: $SHARD_COUNTS"
echo "[INFO] gpus: fixed=[${FIXED_GPUS}] early_a=[${EARLY_A_GPUS}] early_b=[${EARLY_B_GPUS}] resume=$RESUME"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY RUN] plan only, nothing launched"
  exit 0
fi

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi

attack_cmd() {
  local manifest="$1" out="$2" extra=("${@:3}")
  "$PY" scripts/run_attack.py \
    --config "$CONFIG" --model-type phi4_mm \
    --manifest "$manifest" --limit "$(wc -l < "$manifest" | tr -d ' ')" \
    --steps "$STEPS" --methods "$METHODS" --target "$TARGET" \
    --seed "$SEED" --max-seconds 20 --momentum "$MOMENTUM" \
    --device cuda:0 $RESUME_ARG --out-dir "$out" "${extra[@]}"
}

run_fixed() {
  local shard="$1" gpu="$2"
  local out="$OUT_ROOT/fixed/fixed_$shard" log="$OUT_ROOT/logs/fixed_$shard.log"
  echo "[FIXED START] shard=$shard gpu=$gpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" attack_cmd \
    "$OUT_ROOT/shards/fixed_$shard.jsonl" "$out" >> "$log" 2>&1
  local rc=$?
  echo "[FIXED ATTACK END] shard=$shard rc=$rc $(date -Is)" | tee -a "$log"
  return "$rc"
}

run_early() {
  local shard="$1" agpu="$2" jgpu="$3"
  local out="$OUT_ROOT/early/early_$shard" log="$OUT_ROOT/logs/early_$shard.log"
  echo "[EARLY START] shard=$shard attack_gpu=$agpu judge_gpu=$jgpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$agpu,$jgpu" attack_cmd \
    "$OUT_ROOT/shards/early_$shard.jsonl" "$out" \
    --judge-device cuda:1 --judge-model "$JUDGE_MODEL" \
    --eval-every "$EVAL_EVERY" --early-stop >> "$log" 2>&1
  local rc=$?
  echo "[EARLY ATTACK END] shard=$shard rc=$rc $(date -Is)" | tee -a "$log"
  return "$rc"
}

judge_fixed() {
  local shard="$1" gpu="$2"
  local out="$OUT_ROOT/fixed/fixed_$shard" log="$OUT_ROOT/logs/fixed_judge_$shard.log"
  if [ ! -f "$out/run_index.jsonl" ]; then
    echo "[FIXED JUDGE SKIP] shard=$shard: no run_index.jsonl" | tee -a "$log"
    return
  fi
  echo "[FIXED JUDGE START] shard=$shard gpu=$gpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" --out "$out/judged.jsonl" \
    --response-key phi4_response >> "$log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" --out "$out/judged_original.jsonl" \
    --response-key original_response >> "$log" 2>&1
  "$PY" scripts/summarize_attack.py \
    --judged "$out/judged.jsonl" --original-judged "$out/judged_original.jsonl" \
    --out "$out/summary.json" >> "$log" 2>&1
  echo "[FIXED JUDGE END] shard=$shard $(date -Is)" | tee -a "$log"
}

early_post() {
  local shard="$1" jgpu="$2"
  local out="$OUT_ROOT/early/early_$shard" log="$OUT_ROOT/logs/early_post_$shard.log"
  if [ ! -f "$out/run_index.jsonl" ]; then
    echo "[EARLY POST SKIP] shard=$shard: no run_index.jsonl" | tee -a "$log"
    return
  fi
  echo "[EARLY POST START] shard=$shard judge_gpu=$jgpu $(date -Is)" | tee -a "$log"
  "$PY" scripts/summarize_early_stop.py \
    --input "$out/run_index.jsonl" --steps "$STEPS" \
    --out "$out/summary_early.json" >> "$log" 2>&1
  CUDA_VISIBLE_DEVICES="$jgpu" "$PY" scripts/judge_attack.py \
    --input "$out/run_index.jsonl" --out "$out/judged_checkpoint.jsonl" \
    --response-key phi4_response >> "$log" 2>&1
  "$PY" scripts/summarize_attack.py \
    --judged "$out/judged_checkpoint.jsonl" \
    --out "$out/summary_checkpoint.json" >> "$log" 2>&1
  echo "[EARLY POST END] shard=$shard $(date -Is)" | tee -a "$log"
}

# ------------------------------------------- phase 1: fixed workers + early A --
FIXED_PIDS=()
for i in $(seq 0 $((FIXED_SHARDS - 1))); do
  run_fixed "$i" "${FIXED_ARR[$i]}" &
  FIXED_PIDS+=("$!")
done
run_early "a" "${EA_ARR[0]}" "${EA_ARR[1]}" &
EARLY_A_PID=$!
wait "${FIXED_PIDS[@]}"

# ---------------------------------- fixed offline judge (per shard, parallel) --
JUDGE_PIDS=()
for i in $(seq 0 $((FIXED_SHARDS - 1))); do
  judge_fixed "$i" "${FIXED_ARR[$i]}" &
  JUDGE_PIDS+=("$!")
done
wait "${JUDGE_PIDS[@]}"

# ------------------------------------- phase 2: early B on the freed GPUs -----
if [ "$FIXED_SHARDS" -ge 2 ]; then
  run_early "b" "${EB_ARR[0]}" "${EB_ARR[1]}" &
  EARLY_B_PID=$!
fi
wait "$EARLY_A_PID"
if [ -n "${EARLY_B_PID:-}" ]; then wait "$EARLY_B_PID"; fi

# ----------------------------------------------------- early post processing --
POST_PIDS=()
early_post "a" "${EA_ARR[1]}" &
POST_PIDS+=("$!")
if [ "$FIXED_SHARDS" -ge 2 ]; then
  early_post "b" "${EB_ARR[1]}" &
  POST_PIDS+=("$!")
fi
wait "${POST_PIDS[@]}"

# ----------------------------------------------------------- merge + summary --
: > "$OUT_ROOT/run_index.jsonl"
for f in "$OUT_ROOT"/fixed/*/run_index.jsonl "$OUT_ROOT"/early/*/run_index.jsonl; do
  [ -f "$f" ] && cat "$f" >> "$OUT_ROOT/run_index.jsonl"
done

cat "$OUT_ROOT"/fixed/*/judged.jsonl > "$OUT_ROOT/fixed/judged_all.jsonl" 2>/dev/null || true
cat "$OUT_ROOT"/fixed/*/judged_original.jsonl > "$OUT_ROOT/fixed/judged_original_all.jsonl" 2>/dev/null || true
if [ -s "$OUT_ROOT/fixed/judged_all.jsonl" ]; then
  "$PY" scripts/summarize_attack.py \
    --judged "$OUT_ROOT/fixed/judged_all.jsonl" \
    --original-judged "$OUT_ROOT/fixed/judged_original_all.jsonl" \
    --out "$OUT_ROOT/summary_fixed.json"
fi

cat "$OUT_ROOT"/early/*/run_index.jsonl > "$OUT_ROOT/early/run_index_all.jsonl" 2>/dev/null || true
cat "$OUT_ROOT"/early/*/judged_checkpoint.jsonl > "$OUT_ROOT/early/judged_checkpoint_all.jsonl" 2>/dev/null || true
if [ -s "$OUT_ROOT/early/run_index_all.jsonl" ]; then
  "$PY" scripts/summarize_early_stop.py \
    --input "$OUT_ROOT/early/run_index_all.jsonl" --steps "$STEPS" \
    --out "$OUT_ROOT/summary_early.json"
fi
if [ -s "$OUT_ROOT/early/judged_checkpoint_all.jsonl" ]; then
  "$PY" scripts/summarize_attack.py \
    --judged "$OUT_ROOT/early/judged_checkpoint_all.jsonl" \
    --out "$OUT_ROOT/summary_checkpoint.json"
fi

"$PY" - "$OUT_ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}
fixed = load(os.path.join(root, "summary_fixed.json"))
early = load(os.path.join(root, "summary_early.json"))
ckpt = load(os.path.join(root, "summary_checkpoint.json"))
methods = sorted(set(fixed) | set(early) | set(ckpt))
agg = {"fixed": {}, "early": {}, "checkpoint": {}}
print("\n=== FIXED-100 (llamaguard ASR) ===")
for m in methods:
    row = fixed.get(m)
    if not row: continue
    agg["fixed"][m] = {"llamaguard_asr": row.get("llamaguard_asr"),
                       "n": row.get("n"),
                       "mean_final_base_loss": row.get("mean_final_base_loss"),
                       "mean_final_internal_loss": row.get("mean_final_internal_loss")}
    print(f"{m:14s} n={row.get('n')} ASR={row.get('llamaguard_asr')} "
          f"base={row.get('mean_final_base_loss')}")
print("\n=== EARLY-STOP ===")
for m in methods:
    row = early.get(m)
    if not row: continue
    agg["early"][m] = {"asr": row.get("asr"), "n": row.get("n"),
                       "mean_success_step": row.get("mean_success_step"),
                       "mean_evaluated_checkpoints": row.get("mean_evaluated_checkpoints")}
    print(f"{m:14s} n={row.get('n')} ASR={row.get('asr')} "
          f"success_step={row.get('mean_success_step')}")
print("\n=== EARLY-STOP CHECKPOINT JUDGE ===")
for m in methods:
    row = ckpt.get(m)
    if not row: continue
    agg["checkpoint"][m] = {"llamaguard_asr": row.get("llamaguard_asr"),
                            "n": row.get("n")}
    print(f"{m:14s} n={row.get('n')} ASR={row.get('llamaguard_asr')}")
json.dump(agg, open(os.path.join(root, "aggregate.json"), "w"),
          ensure_ascii=False, indent=2)
print(f"\n[OK] {os.path.join(root, 'aggregate.json')}")
PY

echo "[ALL DONE] $(date -Is)"
echo "[INFO] results:"
echo "  $OUT_ROOT/summary_fixed.json"
echo "  $OUT_ROOT/summary_early.json"
echo "  $OUT_ROOT/summary_checkpoint.json"
echo "  $OUT_ROOT/aggregate.json"

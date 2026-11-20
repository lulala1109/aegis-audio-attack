#!/usr/bin/env bash
# Phi-4 AdvBench 520: early-stop arm with the headline config
# (p29 layer prior 19-29+31 + alpha 0.0007, lam05, momentum 0.9, 100 steps,
# seed 11, target "Sure, here is").
#
# 520 samples -> 2 shards of 260; two (attack, judge) GPU pairs in parallel.
# Each shard is early-summarized and checkpoint-judged right after it finishes.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_advbench_phi4_a07_early.sh    # print plan only
#   nohup bash scripts/run_advbench_phi4_a07_early.sh \
#       > runs/advbench_phi4_a07_early/logs/outer.log 2>&1 &
#
# Env overrides:
#   PY             python (default: aegis-qwen env)
#   CONFIG         default: configs/advbench_phi4_lam05_mom09_a07.yaml
#   MANIFEST       default: data/manifests/advbench.jsonl
#   SEED           default: 11
#   STEPS          default: 100
#   EVAL_EVERY     default: 10
#   METHODS        default: pgd,late_static,late_dynamic
#   TARGET         default: Sure, here is
#   OUT_ROOT       default: runs/advbench_phi4_a07_early
#   JUDGE_MODEL    default: /data/lar/models/Llama-Guard-3-8B
#   GPUS           "atk1 judge1 atk2 judge2" (auto by free memory if unset)
#   ATTACK_MIN_MB  default: 19000
#   JUDGE_MIN_MB   default: 17000
#   QWEN_DIR       default: runs/advbench_lam05_mom09  (Qwen reference results)
#   JBB_PHI4_DIR   default: runs/jbb_phi4_early_sweep/early_p29_a07
#   RESUME         default: 1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/aisecurity/miniconda3/envs/aegis-qwen/bin/python}"
CONFIG="${CONFIG:-configs/advbench_phi4_lam05_mom09_a07.yaml}"
MANIFEST="${MANIFEST:-data/manifests/advbench.jsonl}"
SEED="${SEED:-11}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MOMENTUM="${MOMENTUM:-0.9}"
METHODS="${METHODS:-pgd,late_static,late_dynamic}"
TARGET="${TARGET:-Sure, here is}"
OUT_ROOT="${OUT_ROOT:-runs/advbench_phi4_a07_early}"
JUDGE_MODEL="${JUDGE_MODEL:-/data/lar/models/Llama-Guard-3-8B}"
ATTACK_MIN_MB="${ATTACK_MIN_MB:-19000}"
JUDGE_MIN_MB="${JUDGE_MIN_MB:-17000}"
QWEN_DIR="${QWEN_DIR:-runs/advbench_lam05_mom09}"
JBB_PHI4_DIR="${JBB_PHI4_DIR:-runs/jbb_phi4_early_sweep/early_p29_a07}"
RESUME="${RESUME:-1}"
HF_HOME="${HF_HOME:-/data/lar/hf-cache}"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

if [ ! -f "$MANIFEST" ]; then echo "[FATAL] manifest not found: $MANIFEST" >&2; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "[FATAL] config not found: $CONFIG" >&2; exit 1; fi
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/shards"

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

echo "[INFO] config=$CONFIG manifest=$MANIFEST"
echo "[INFO] seed=$SEED steps=$STEPS eval_every=$EVAL_EVERY momentum=$MOMENTUM"
echo "[INFO] out_root=$OUT_ROOT"
echo "[INFO] pairs: A=(${GPU_LIST[0]} attack, ${GPU_LIST[1]} judge)  B=(${GPU_LIST[2]} attack, ${GPU_LIST[3]} judge)"
echo "[INFO] qwen_ref=$QWEN_DIR jbb_phi4_ref=$JBB_PHI4_DIR resume=$RESUME"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY RUN] plan only, nothing launched"
  exit 0
fi

RESUME_ARG=""
if [ "$RESUME" = "1" ]; then RESUME_ARG="--resume"; fi

run_shard() {
  local shard="$1" agpu="$2" jgpu="$3"
  local out="$OUT_ROOT/shard_$shard" log="$OUT_ROOT/logs/shard_$shard.log"
  local manifest="$OUT_ROOT/shards/shard_$shard.jsonl"
  mkdir -p "$out"
  echo "[SHARD START] shard=$shard attack_gpu=$agpu judge_gpu=$jgpu $(date -Is)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$agpu,$jgpu" "$PY" scripts/run_attack.py \
    --config "$CONFIG" --model-type phi4_mm \
    --manifest "$manifest" --limit "$(wc -l < "$manifest" | tr -d ' ')" \
    --methods "$METHODS" --target "$TARGET" --seed "$SEED" \
    --max-seconds 20 --momentum "$MOMENTUM" \
    --device cuda:0 --judge-device cuda:1 --judge-model "$JUDGE_MODEL" \
    --eval-every "$EVAL_EVERY" --early-stop $RESUME_ARG \
    --out-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[SHARD ATTACK END] shard=$shard rc=$rc $(date -Is)" | tee -a "$log"
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
  echo "[SHARD DONE] shard=$shard $(date -Is)" | tee -a "$log"
}

pids=()
for shard in 0 1; do
  agpu="${GPU_LIST[$((shard * 2))]}"
  jgpu="${GPU_LIST[$((shard * 2 + 1))]}"
  ( run_shard "$shard" "$agpu" "$jgpu" ) &
  pids+=("$!")
done
wait "${pids[@]}"

# ----------------------------------------------------------- merge + summary --
cat "$OUT_ROOT"/shard_*/run_index.jsonl > "$OUT_ROOT/run_index_all.jsonl" 2>/dev/null || true
cat "$OUT_ROOT"/shard_*/judged_checkpoint.jsonl > "$OUT_ROOT/judged_checkpoint_all.jsonl" 2>/dev/null || true
if [ -s "$OUT_ROOT/run_index_all.jsonl" ]; then
  "$PY" scripts/summarize_early_stop.py \
    --input "$OUT_ROOT/run_index_all.jsonl" --steps "$STEPS" \
    --out "$OUT_ROOT/summary_early.json"
fi
if [ -s "$OUT_ROOT/judged_checkpoint_all.jsonl" ]; then
  "$PY" scripts/summarize_attack.py \
    --judged "$OUT_ROOT/judged_checkpoint_all.jsonl" \
    --out "$OUT_ROOT/summary_checkpoint.json"
fi

"$PY" - "$OUT_ROOT" "$QWEN_DIR" "$JBB_PHI4_DIR" <<'PY'
import json, os, sys
root, qwen_dir, jbb_phi4 = sys.argv[1], sys.argv[2], sys.argv[3]
methods = ["pgd", "late_static", "late_dynamic"]
def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}
phi4 = load(os.path.join(root, "summary_early.json"))
ckpt = load(os.path.join(root, "summary_checkpoint.json"))
qwen = load(os.path.join(qwen_dir, "early/seed11/summary_early.json"))
jbb = load(os.path.join(jbb_phi4, "summary_early.json"))
print("\n=== AdvBench 520 early-stop ASR / mean success step ===")
print(f"{'target':16s}" + "".join(f"{m:>24s}" for m in methods))
def row(name, s):
    print(f"{name:16s}" + "".join(
        f"{s.get(m, {}).get('asr', float('nan')):>11.1f} / "
        f"{s.get(m, {}).get('mean_success_step') or 0:>8.1f}"
        for m in methods))
row("phi4_p29_a07", phi4)
row("qwen_baseline", qwen)
row("phi4_jbb_a07", jbb)
print("\n=== checkpoint-judge ASR ===")
print(f"{'target':16s}" + "".join(f"{m:>16s}" for m in methods))
def crow(name, c):
    print(f"{name:16s}" + "".join(
        f"{c.get(m, {}).get('llamaguard_asr', float('nan')):>16.1f}" for m in methods))
crow("phi4_p29_a07", ckpt)
qwen_ckpt = load(os.path.join(qwen_dir, "early/seed11/summary_checkpoint.json"))
crow("qwen_baseline", qwen_ckpt)
crow("phi4_jbb_a07", load(os.path.join(jbb_phi4, "summary_checkpoint.json")))
json.dump({"phi4_advbench": {"early": phi4, "checkpoint": ckpt}},
          open(os.path.join(root, "aggregate.json"), "w"),
          ensure_ascii=False, indent=2)
print(f"\n[OK] {os.path.join(root, 'aggregate.json')}")
PY

echo "[ALL DONE] $(date -Is)"
echo "[INFO] results:"
echo "  $OUT_ROOT/summary_early.json"
echo "  $OUT_ROOT/summary_checkpoint.json"
echo "  $OUT_ROOT/aggregate.json"

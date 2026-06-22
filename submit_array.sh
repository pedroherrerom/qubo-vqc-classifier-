#!/usr/bin/env bash
# submit_array.sh — HPC job-array submission for CESGA Finisterrae III.
#
# Spreads independent VQC runs across SLURM array tasks, then merges all
# outputs with a gated aggregation job.
#
# Usage:
#   bash submit_array.sh --config configVQC.json [options]
#
# Options:
#   --config           Path to JSON config file           [configVQC.json]
#   --num-runs         Total independent runs             [from config]
#   --runs-per-task    Runs per array task                [1]
#   --max-concurrent   Max simultaneous tasks (0=no cap)  [0]
#   --vqc-time         Wall-clock limit per task          [04:00:00]
#   --vqc-mem          Memory per task                    [8G]
#   --agg-time         Aggregation job wall-clock         [01:00:00]

set -euo pipefail

CONFIG="configVQC.json"
NUM_RUNS=""
RUNS_PER_TASK=1
MAX_CONCURRENT=0
VQC_TIME="04:00:00"
VQC_MEM="8G"
AGG_TIME="01:00:00"

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="$2";          shift 2 ;;
        --num-runs)        NUM_RUNS="$2";         shift 2 ;;
        --runs-per-task)   RUNS_PER_TASK="$2";   shift 2 ;;
        --max-concurrent)  MAX_CONCURRENT="$2";  shift 2 ;;
        --vqc-time)        VQC_TIME="$2";         shift 2 ;;
        --vqc-mem)         VQC_MEM="$2";          shift 2 ;;
        --agg-time)        AGG_TIME="$2";          shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- read num_runs from config if not overridden ----------
if [[ -z "$NUM_RUNS" ]]; then
    NUM_RUNS=$(python - <<EOF
import json
with open("$CONFIG") as f:
    cfg = json.load(f)
print(cfg.get("num_runs", 5))
EOF
)
fi

# ---------- compute array size ----------
N_TASKS=$(( (NUM_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
[[ "$MAX_CONCURRENT" -gt 0 ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"

# ---------- pre-create experiment directory on login node ----------
EXP_DIR=$(python make_exp_dir.py --config "$CONFIG")
echo "=== Submitting job array ==="
echo "  Config      : $CONFIG"
echo "  Experiment  : $EXP_DIR"
echo "  Runs        : $NUM_RUNS  (runs_per_task=$RUNS_PER_TASK)"
echo "  Array       : $ARRAY_SPEC"
echo "  VQC time    : $VQC_TIME  mem: $VQC_MEM"
echo "  Agg time    : $AGG_TIME"

mkdir -p logs

# ---------- submit array job ----------
ARRAY_JID=$(sbatch \
    --parsable \
    --array="$ARRAY_SPEC" \
    --time="$VQC_TIME" \
    --mem="$VQC_MEM" \
    --output="logs/vqc_array-%A_%a.out" \
    --error="logs/vqc_array-%A_%a.err" \
    vqc_array_job.sh "$CONFIG" "$EXP_DIR" "$RUNS_PER_TASK")

echo "  Array job   : $ARRAY_JID"

# ---------- submit aggregation job (runs after ALL array tasks) ----------
AGG_JID=$(sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JID}" \
    --time="$AGG_TIME" \
    --mem="4G" \
    --output="logs/vqc_aggregate-%j.out" \
    --error="logs/vqc_aggregate-%j.err" \
    aggregate_job.sh "$CONFIG" "$EXP_DIR")

echo "  Agg job     : $AGG_JID"
echo ""
echo "Monitor with:  watch -n 10 squeue --me"
echo "Array logs:    tail -f logs/vqc_array-${ARRAY_JID}_0.out"
echo "Agg logs:      tail -f logs/vqc_aggregate-${AGG_JID}.out"

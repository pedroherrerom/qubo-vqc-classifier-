#!/usr/bin/env bash
# vqc_array_job.sh — SLURM array task: execute one chunk of independent runs.
#
# Each task handles RUNS_PER_TASK independent VQC runs identified by
# SLURM_ARRAY_TASK_ID * RUNS_PER_TASK … (SLURM_ARRAY_TASK_ID+1)*RUNS_PER_TASK-1.
#
# Called by submit_array.sh with:
#   sbatch --array=... vqc_array_job.sh <config> <exp_dir> <runs_per_task>
#
# SLURM directives below are placeholders — adjust for your allocation.

#SBATCH --job-name=vqc_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/vqc_array-%A_%a.out
#SBATCH --error=logs/vqc_array-%A_%a.err

set -euo pipefail

CONFIG="${1:-configVQC.json}"
EXP_DIR="${2:?EXP_DIR argument required}"
RUNS_PER_TASK="${3:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Load cluster modules
module load gcc/14.3.0
module load openmpi/5.0.9
module load rust/1.88.0

echo "[task $TASK_ID] Starting at $(date)"
echo "[task $TASK_ID] EXP_DIR=$EXP_DIR  runs_per_task=$RUNS_PER_TASK"

python main_VQC.py \
    --config "$CONFIG" \
    --mode task \
    --exp-dir "$EXP_DIR" \
    --task-id "$TASK_ID" \
    --runs-per-task "$RUNS_PER_TASK"

echo "[task $TASK_ID] Done at $(date)"

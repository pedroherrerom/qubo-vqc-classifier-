#!/usr/bin/env bash
# aggregate_job.sh — SLURM aggregation job.
#
# Merges per-task outputs from tasks/ into the experiment root and generates
# all final plots. Triggered only after ALL array tasks complete successfully
# (dependency: afterok:<ARRAY_JID>).
#
# Called by submit_array.sh with:
#   sbatch aggregate_job.sh <config> <exp_dir>
#
# SLURM directives below are placeholders — adjust for your allocation.

#SBATCH --job-name=vqc_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/vqc_aggregate-%j.out
#SBATCH --error=logs/vqc_aggregate-%j.err

set -euo pipefail

CONFIG="${1:-configVQC.json}"
EXP_DIR="${2:?EXP_DIR argument required}"

# Load cluster modules
module load gcc/14.3.0

echo "[aggregate] Starting at $(date)"
echo "[aggregate] EXP_DIR=$EXP_DIR"

python main_VQC.py \
    --config "$CONFIG" \
    --mode aggregate \
    --exp-dir "$EXP_DIR"

echo "[aggregate] Done at $(date)"
echo "[aggregate] Results → $EXP_DIR"

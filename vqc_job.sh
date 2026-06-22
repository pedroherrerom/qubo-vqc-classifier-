#!/usr/bin/env bash
# vqc_job.sh — SLURM job: run the VQC pipeline (depends on qraise_job when
# vqc_test_infrastructure=cunqa).
#
# Called by submit.sh with:
#   sbatch vqc_job.sh <config> [extra args...]
#
# SLURM directives below are placeholders — adjust for your allocation.

#SBATCH --job-name=vqc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/vqc-%j.out
#SBATCH --error=logs/vqc-%j.err

set -euo pipefail

CONFIG="${1:-configVQC.json}"
shift || true   # remaining args forwarded to main_VQC.py

# Load cluster modules
module load gcc/14.3.0
module load openmpi/5.0.9
module load rust/1.88.0

echo "[vqc_job] Starting at $(date) — config: $CONFIG"

python main_VQC.py --config "$CONFIG" "$@"

echo "[vqc_job] Done at $(date)"

#!/usr/bin/env bash
# qraise_job.sh — SLURM job: provision CUNQA QPUs on CESGA Finisterrae III.
#
# This job runs before the VQC job and keeps QPUs alive for the duration
# of the experiment. The VQC job depends on this one via --dependency=after.
#
# SLURM directives below are placeholders — adjust for your allocation.

#SBATCH --job-name=qraise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/qraise-%j.out
#SBATCH --error=logs/qraise-%j.err

set -euo pipefail

# Load required cluster modules
module load gcc/14.3.0
module load openmpi/5.0.9

echo "[qraise] Starting CUNQA QPU provisioning at $(date)"

# Raise QPUs (adjust --num-qpus and --qubits-per-qpu as needed)
qraise --num-qpus 4 --qubits-per-qpu 8 --backend aer

echo "[qraise] QPUs ready at $(date)"

#!/usr/bin/env bash
# submit.sh — HPC single-job submission for CESGA Finisterrae III.
#
# Chains two SLURM jobs:
#   1. qraise_job.sh  — provision CUNQA QPUs
#   2. vqc_job.sh     — run the VQC pipeline (depends on qraise)
#
# Usage:
#   bash submit.sh --config configVQC.json [options]
#
# Options:
#   --config            Path to JSON config file           [configVQC.json]
#   --vqc-n-workers     PSO parallel workers               [from config]
#   --vqc-cores-per-worker  CPU cores per worker           [from config]
#   --vqc-time          Wall-clock limit for VQC job       [08:00:00]
#   --vqc-mem           Memory per VQC job                 [8G]
#   --qraise-time       Wall-clock limit for qraise job    [VQC time + 1h]

set -euo pipefail

CONFIG="configVQC.json"
VQC_TIME="08:00:00"
VQC_MEM="8G"
VQC_N_WORKERS=""
VQC_CORES=""
QRAISE_TIME=""

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)              CONFIG="$2";         shift 2 ;;
        --vqc-time)            VQC_TIME="$2";       shift 2 ;;
        --vqc-mem)             VQC_MEM="$2";        shift 2 ;;
        --vqc-n-workers)       VQC_N_WORKERS="$2";  shift 2 ;;
        --vqc-cores-per-worker) VQC_CORES="$2";     shift 2 ;;
        --qraise-time)         QRAISE_TIME="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- derive qraise time (VQC time + 1h) ----------
if [[ -z "$QRAISE_TIME" ]]; then
    IFS=':' read -r hh mm ss <<< "$VQC_TIME"
    hh=$(( 10#$hh + 1 ))
    QRAISE_TIME=$(printf "%02d:%s:%s" "$hh" "$mm" "$ss")
fi

# ---------- optional CLI overrides ----------
EXTRA_ARGS=""
[[ -n "$VQC_N_WORKERS" ]] && EXTRA_ARGS="$EXTRA_ARGS --vqc-n-workers $VQC_N_WORKERS"
[[ -n "$VQC_CORES"     ]] && EXTRA_ARGS="$EXTRA_ARGS --vqc-cores-per-worker $VQC_CORES"

echo "=== Submitting single-job VQC pipeline ==="
echo "  Config      : $CONFIG"
echo "  VQC time    : $VQC_TIME  mem: $VQC_MEM"
echo "  qraise time : $QRAISE_TIME"

# ---------- read test infrastructure from config ----------
TEST_INFRA=$(python - <<EOF
import json, sys
with open("$CONFIG") as f:
    cfg = json.load(f)
print(cfg.get("vqc_test_infrastructure", "local"))
EOF
)

# ---------- submit jobs ----------
mkdir -p logs

if [[ "$TEST_INFRA" == "cunqa" ]]; then
    # Submit qraise first, then VQC with dependency
    QRAISE_JID=$(sbatch \
        --parsable \
        --time="$QRAISE_TIME" \
        --output="logs/qraise-%j.out" \
        --error="logs/qraise-%j.err" \
        qraise_job.sh)
    echo "  qraise job  : $QRAISE_JID"

    VQC_JID=$(sbatch \
        --parsable \
        --time="$VQC_TIME" \
        --mem="$VQC_MEM" \
        --dependency="after:$QRAISE_JID" \
        --output="logs/vqc-%j.out" \
        --error="logs/vqc-%j.err" \
        vqc_job.sh "$CONFIG" $EXTRA_ARGS)
else
    # Local infrastructure — no qraise needed
    VQC_JID=$(sbatch \
        --parsable \
        --time="$VQC_TIME" \
        --mem="$VQC_MEM" \
        --output="logs/vqc-%j.out" \
        --error="logs/vqc-%j.err" \
        vqc_job.sh "$CONFIG" $EXTRA_ARGS)
fi

echo "  VQC job     : $VQC_JID"
echo "Done. Monitor with: squeue --me"

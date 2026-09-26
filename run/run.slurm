#!/bin/bash
#SBATCH -J mp_interstitial
#SBATCH -o logs/%j.log
#SBATCH --tasks-per-node=12       # 4 tasks per GPU x 3 GPUs
#SBATCH -p gpu-a100
#SBATCH -t 23:59:59
#SBATCH -A CHE21006                # <-- set to your actual allocation
#SBATCH -N 1

set -e  # fail fast on setup errors, rather than silently continuing into a broken MPS state

GPUS_PER_NODE=3
TASKS_PER_GPU=4

pwd
hostname -f
date

# =========================================================
# CUDA / MPS SETUP
# =========================================================
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

# Use a job-unique MPS directory so this job never collides with, or picks up,
# a stale daemon/socket left behind by a previous job on the same node.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Kill any leftover MPS control daemon from a prior job that might still be
# running under this user on this node before starting a fresh one.
echo quit | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" nvidia-cuda-mps-control 2>/dev/null || true
pkill -u "$USER" -f nvidia-cuda-mps-control 2>/dev/null || true
sleep 2

# MPS control daemon must see the same GPUs the job will use. On a single
# node this just needs to run once (no srun -N/-n fan-out needed here).
nvidia-cuda-mps-control -d

# Give the daemon a moment to actually come up, then verify it did.
sleep 3
if ! pgrep -u "$USER" -f nvidia-cuda-mps-control > /dev/null; then
    echo "ERROR: nvidia-cuda-mps-control failed to start. Aborting." >&2
    exit 1
fi
echo "MPS control daemon is running (pipe dir: $CUDA_MPS_PIPE_DIRECTORY)"

# Always shut MPS down on exit, whether the job succeeds, fails, or is
# cancelled, so it never leaks into the next job on this node.
cleanup() {
    echo "Shutting down MPS..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT

# =========================================================
# ENVIRONMENT
# =========================================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate conda1
python --version

export FAIRCHEM_CACHE_DIR=${FAIRCHEM_CACHE_DIR:-$SCRATCH/fairchem_cache}

# =========================================================
# RUN JOB — 12 tasks on 1 node, 4 tasks sharing each of 3 GPUs via MPS
# =========================================================
echo "Running $((GPUS_PER_NODE * TASKS_PER_GPU)) tasks across $GPUS_PER_NODE GPUs (MPS)..."

srun -N 1 -n $((GPUS_PER_NODE * TASKS_PER_GPU)) --spread-job \
    bash -c '
        GPUS_PER_NODE='"$GPUS_PER_NODE"'
        CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % GPUS_PER_NODE ))
        export CUDA_VISIBLE_DEVICES

        # Explicitly re-export so every task in the fan-out definitely sees
        # the same MPS pipe/log dirs set up above, rather than relying on
        # srun to have propagated them correctly.
        export CUDA_MPS_PIPE_DIRECTORY='"$CUDA_MPS_PIPE_DIRECTORY"'
        export CUDA_MPS_LOG_DIRECTORY='"$CUDA_MPS_LOG_DIRECTORY"'

        echo "task $SLURM_PROCID (local $SLURM_LOCALID) -> GPU $CUDA_VISIBLE_DEVICES"

        python -u get_data.py
    '

echo "Job done"
date

#!/bin/bash
#SBATCH -J mp_interstitial
#SBATCH -o logs/%j.log
#SBATCH --tasks-per-node=12       # 4 tasks per GPU x 3 GPUs
#SBATCH -p gpu-a100
#SBATCH -t 1:59:00
#SBATCH -A CHE21006                # <-- set to your actual allocation
#SBATCH -N 1

GPUS_PER_NODE=3
TASKS_PER_GPU=4

pwd
hostname -f
date

# =========================================================
# CUDA / MPS SETUP
# =========================================================
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# MPS control daemon must see the same GPUs the job will use. On a single
# node this just needs to run once (no srun -N/-n fan-out needed here).
nvidia-cuda-mps-control -d

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

        echo "task $SLURM_PROCID (local $SLURM_LOCALID) -> GPU $CUDA_VISIBLE_DEVICES"

        python -u get_data.py
    '

# =========================================================
# SHUT DOWN MPS
# =========================================================
echo quit | nvidia-cuda-mps-control

echo "Job done"
date


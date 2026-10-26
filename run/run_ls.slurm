#!/bin/bash
#SBATCH -J mp_interstitial
#SBATCH -o logs/%j.log
#SBATCH --tasks-per-node=12       # 4 tasks per GPU x 3 GPUs
#SBATCH -p gpu-a100
#SBATCH -t 47:59:59
#SBATCH -A CHE21006
#SBATCH -N 1

set -e  # fail fast on setup errors, rather than silently continuing into a broken MPS state

GPUS_PER_NODE=3
TASKS_PER_GPU=4

pwd
hostname -f
date

source $(conda info --base)/etc/profile.d/conda.sh
conda activate conda1
python --version

echo "Running $((GPUS_PER_NODE * TASKS_PER_GPU)) tasks across $GPUS_PER_NODE GPUs (MPS)..."

srun -N 1 -n $((GPUS_PER_NODE * TASKS_PER_GPU)) --spread-job \
    bash -c '
        GPUS_PER_NODE='"$GPUS_PER_NODE"'
        CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % GPUS_PER_NODE ))
        export CUDA_VISIBLE_DEVICES

        echo "task $SLURM_PROCID (local $SLURM_LOCALID) -> GPU $CUDA_VISIBLE_DEVICES"

        python -u get_data.py
    '

echo "Job done"
date

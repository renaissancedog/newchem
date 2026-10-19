#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=li_extract
#SBATCH --gres=gpu:a100:3
#SBATCH --output=run.log
#SBATCH --ntasks=12              # GPUS_PER_NODE * TASKS_PER_GPU
#SBATCH --cpus-per-task=1       # adjust to what get_data.py needs

GPUS_PER_NODE=3
TASKS_PER_GPU=4

pwd
hostname -f
date
source ~/chem/.venv/bin/activate
python --version

echo "Running $((GPUS_PER_NODE * TASKS_PER_GPU)) tasks across $GPUS_PER_NODE GPUs..."

srun -N 1 -n $((GPUS_PER_NODE * TASKS_PER_GPU)) \
     --gres=gpu:a100:$GPUS_PER_NODE --gpu-bind=none --overlap \
    bash -c '
        GPUS_PER_NODE='"$GPUS_PER_NODE"'
        export CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % GPUS_PER_NODE ))

        echo "task $SLURM_PROCID (local $SLURM_LOCALID) -> GPU $CUDA_VISIBLE_DEVICES"

        python -u get_data.py
    '

date

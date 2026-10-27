#!/bin/bash
#SBATCH -J mp_interstitial
#SBATCH -o logs/%j.log
#SBATCH --tasks-per-node=12       # 4 tasks per GPU x 3 GPUs
#SBATCH -p gpu-a100
#SBATCH -t 47:59:59
#SBATCH -A CHE21006
#SBATCH -N 6

NODES=6
GPUS_PER_NODE=3
TASKS_PER_GPU=4

source ~/.bashrc
cd newchem/run
pwd
hostname -f
date
conda activate conda1
python --version

echo "Running $((NODES * GPUS_PER_NODE * TASKS_PER_GPU)) tasks across $((NODES * GPUS_PER_NODE)) GPUs (MPS)..."

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID

srun -N $NODES -n $((NODES * GPUS_PER_NODE * TASKS_PER_GPU)) \
     --ntasks-per-node=$((GPUS_PER_NODE * TASKS_PER_GPU)) \
    bash -c '
        mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
        [ "$SLURM_LOCALID" -eq 0 ] && nvidia-cuda-mps-control -d
        sleep 5

        export CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % '"$GPUS_PER_NODE"' ))
        python -u get_data.py
    '

echo "Job done"
date

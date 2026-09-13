#!/bin/bash
#SBATCH --job-name=li_extract
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm.out

pwd
hostname -f
date
python --version

source $HOME/chem/.venv/bin/activate
python get_data.py

echo "Job done"
date

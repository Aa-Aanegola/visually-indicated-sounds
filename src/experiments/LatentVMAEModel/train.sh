#!/bin/bash
#SBATCH -J vis_indicated
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -w gnode056
#SBATCH --mem=25G
#SBATCH --time=4-00:00:00
#SBATCH -o slurm_logs/audio_conv.out

export PYTHONUNBUFFERED=1

python3 TrainVISLatentVMAE.py

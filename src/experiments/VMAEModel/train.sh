#!/bin/bash
#SBATCH -J vis_indicated
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 25
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH -o slurm_logs/vmae_plaf.out

../../../fetch_train.sh v2
../../../fetch_test.sh v2

python TrainVMAEModel.py

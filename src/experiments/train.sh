#!/bin/bash
#SBATCH -J vis_indicated
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4-00:00:00
#SBATCH -o train_log_loss_2.out

source $CONDA_PREFIX/bin/activate /home2/arihanth.srikar/miniconda3/envs/torch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export PYTHONUNBUFFERED=1

bash ../../fetch_test.sh
bash ../../fetch_train.sh
python train_model.py
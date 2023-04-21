#!/bin/bash
#SBATCH -J vis_indicated
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH -w gnode051
#SBATCH --mem=25G

# source $CONDA_PREFIX/bin/activate /home2/arihanth.srikar/miniconda3/envs/torch
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export PYTHONUNBUFFERED=1

bash ../../../fetch_test.sh
bash ../../../fetch_train.sh
python TrainPaperModel.py

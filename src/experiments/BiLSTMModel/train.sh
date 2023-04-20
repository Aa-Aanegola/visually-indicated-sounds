#!/bin/bash
#SBATCH -J vis_indicated
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 25
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH -o slurm_logs/mse_bi_plaf.out

# source $CONDA_PREFIX/bin/activate /scratch/arihanth.srikar/pytorch_env
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export PYTHONUNBUFFERED=1

bash ../../../fetch_test.sh
bash ../../../fetch_train.sh

python TrainBiLSTMModel.py
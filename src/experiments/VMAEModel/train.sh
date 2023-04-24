#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH -o output.txt
#SBATCH --job-name=vis-vmae
#SBATCH -w gnode057

../../../fetch_train.sh v2
../../../fetch_test.sh v2

python TrainVMAEModel.py

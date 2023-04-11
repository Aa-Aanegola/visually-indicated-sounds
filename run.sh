#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --mem=30G
#SBATCH -o output.txt
#SBATCH --job-name=preprocess-data

mkdir -p /scratch/aanegola
cp -r ~/visually-indicated-sounds /scratch/aanegola/

cd /scratch/aanegola/visually-indicated-sounds/src/experiments
python3 preprocess_data.py
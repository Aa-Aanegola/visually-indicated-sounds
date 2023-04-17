#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --mem=30G
#SBATCH -o output.txt
#SBATCH --job-name=preprocess-data


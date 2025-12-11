#!/bin/bash
SBATCH --job-name=train_128
SBATCH --output=train_128_%j.out
SBATCH --gres=gpu:v100-sxm2:1
SBATCH --cpus-per-task=8
SBATCH --mem=32GB
SBATCH --time=04:00:00
SBATCH --partition=gpu

module load cuda/12.3.0 python/3.13.5
source ~/venvs/diffusion/bin/activate

cd ~/text-to-image-flickr8k/"Assignemnt 2"
python train_diffusion.py
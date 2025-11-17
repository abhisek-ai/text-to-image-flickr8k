#!/bin/bash
#SBATCH --job-name=diff_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="teslap100|v100"
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --partition=gpu

echo "===================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting: $(date)"
echo "===================="

# Load modules
module load cuda/11.8
module load python/3.10

# Activate venv
source ~/venvs/diffusion/bin/activate

# Show GPU info
echo "GPU Information:"
nvidia-smi

echo ""
echo "Starting training..."
echo ""

# Run training
python train_diffusion.py

echo ""
echo "===================="
echo "Finished: $(date)"
echo "===================="
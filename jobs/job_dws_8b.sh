#!/bin/bash
#SBATCH --job-name=olala-8b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/olala_8b_%j.out
#SBATCH --error=logs/olala_8b_%j.err

set -euo pipefail

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

# Load environment
set -a
source .env.dws
set +a

python run_experiment.py --model "$MODEL_PATH" --wandb --threshold 0.6

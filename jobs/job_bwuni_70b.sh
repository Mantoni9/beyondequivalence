#!/bin/bash
#SBATCH --job-name=olala-70b
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/olala_70b_%j.out
#SBATCH --error=logs/olala_70b_%j.err

set -euo pipefail

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

# Load environment
set -a
source .env.bwuni
set +a

python run_experiment.py --model "$MODEL_PATH" --wandb --threshold 0.6

#!/bin/bash
#SBATCH --job-name=olala_dws_70b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/olala_dws_70b_%j.out
#SBATCH --error=logs/olala_dws_70b_%j.err

set -euo pipefail

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

# Load environment
set -a
source .env.dws
set +a

# Enable 8-bit quantization for the 70B model on 2x A6000 (96 GB VRAM total)
export LOAD_IN_8BIT=true

python run_experiment.py --model "$MODEL_PATH" --wandb --threshold 0.6

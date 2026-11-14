#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=dual_gate
#SBATCH --output=%x_%j.out
# Dual-threshold gating (Jeroen's recommended extension). Baselines are cached.
# (1) per_domain: grid around the paper's zero-shot thresholds (Bio 0.6, Cyber 0.2); (2) any_all at the paper values.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/dual_gate.py --modes per_domain --tau-bio 0.5 0.6 0.7 --tau-cyber 0.15 0.2 0.25 0.3 --tag per_domain
python zero_shot/dual_gate.py --modes any_all --tau-bio 0.6 --tau-cyber 0.2 --tag any_all
echo ALL_STEPS_DONE

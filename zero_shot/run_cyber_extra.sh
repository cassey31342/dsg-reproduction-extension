#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_cyx
#SBATCH --output=%x_%j.out
# Cyber features: fill in the paper's tau grid (0.05, 0.15 were missing) and add the cross-domain datasets Jeroen suggested
# (MMLU computer science) plus WMDP-Bio. Baselines for these datasets are cached, so this takes minutes.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/run_zero_shot.py --target cyber --taus 0.05 0.1 0.15 0.2 0.25 0.3 --extra-datasets college_computer_science wmdp-bio --tag cyber_grid_cross

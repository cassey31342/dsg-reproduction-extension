#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_multi
#SBATCH --output=%x_%j.out
# Figure 4 style multi-task: Bio + Cyber features together (40), shared tau sweep. Baselines are cached.
# Second command completes the Cyber-only sweep with WMDP-Bio and college computer science for a fair 3-way comparison.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/run_zero_shot.py --target both --taus 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --tag multi_union
python zero_shot/run_zero_shot.py --target cyber --taus 0.4 0.5 0.6 0.7 0.8 0.9 --extra-datasets wmdp-bio college_computer_science --tag cross_rest

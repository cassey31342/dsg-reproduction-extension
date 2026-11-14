#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_smoke
#SBATCH --output=%x_%j.out
# Timing run: bio features, tau 0.6, ONE small MMLU dataset. Purpose: does the script run, and how long is a step.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/run_zero_shot.py --target bio --datasets college_computer_science --tag smoke

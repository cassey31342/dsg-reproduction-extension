#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gate_stats
#SBATCH --output=%x_%j.out
# Per-prompt gate statistics rho_i(x). First argument: extra args for gate_stats.py (e.g. "--tag smoke --sets wiki_test_1024 wmdp-bio mcq_cal")
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/gate_stats.py "$@"
echo ALL_STEPS_DONE

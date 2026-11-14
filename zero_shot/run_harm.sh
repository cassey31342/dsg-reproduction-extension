#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_harm
#SBATCH --output=%x_%j.out
# Jeroen's suggestion: zero-shot with harm words (Malicious, Dangerous, Harmful, and their union) instead of topic words.
# Evaluated on WMDP-Bio, WMDP-Cyber and 5 MMLU subjects; tau sweep 0.1..0.9. Baselines are cached.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
for K in harm_malicious harm_dangerous harm_harmful harm_union; do
  echo "##### $K"
  python zero_shot/run_zero_shot.py --target both --features-key $K --taus 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --tag $K || echo "$K FAILED"
done
echo ALL_STEPS_DONE

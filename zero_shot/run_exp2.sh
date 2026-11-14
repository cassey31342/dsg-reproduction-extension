#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp2
#SBATCH --output=%x_%j.out
# Experiment 2: feature overlap. Data-driven top-k features (SAE l0 59), k = 20, 50, 100, 200.
# Per k: full lists vs shared features removed (each with its own calibrated thresholds) vs one shared gate over the union.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/gate_stats.py --feature-keys bio cyber --groups-json results/dd_l059/exp2_groups.json --tag dd \
  --sets wiki_test_1024 wiki_val_1024 wiki_test_128 wiki_val_128 mcq_cal mcq_eval wmdp-bio wmdp-cyber || echo "STEP 1 FAILED"
python zero_shot/exp2_plan.py results/gate_stats_dd.npz results/dd_l059/feature_lists.json results/exp2 || echo "STEP 2 FAILED"
while read -r CMD; do echo "##### $CMD"; $CMD || echo "COMMAND FAILED"; done < results/exp2/commands.sh
echo ALL_STEPS_DONE

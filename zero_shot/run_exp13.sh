#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=exp13
#SBATCH --output=%x_%j.out
# Experiment 1 (calibrated thresholds) and 3 (prompt transfer), zero-shot Bio and Cyber features, SAE l0 59.
#  (a) gate statistics on all prompt sets  (b) thresholds from them  (c) dual gate with each threshold setting  (d) prompt-variant transfer
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/gate_stats.py --tag zs || echo "STEP a FAILED"
python zero_shot/calibrate_gates.py results/gate_stats_zs.npz --out results/gate_calibration_zs.json || echo "STEP b FAILED"
PAIRS=$(python zero_shot/pairs_from_calibration.py results/gate_calibration_zs.json)
echo "threshold settings: $PAIRS"
python zero_shot/transfer.py --configs $PAIRS --tag zs || echo "STEP d FAILED"
python zero_shot/dual_gate.py --modes per_domain --pairs $(python zero_shot/pairs_from_calibration.py results/gate_calibration_zs.json --pairs-only) --tag calibrated || echo "STEP c FAILED"
echo ALL_STEPS_DONE

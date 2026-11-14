#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_sweep
#SBATCH --output=%x_%j.out
# tau sweep 0.1..0.9 (paper Fig. 6B). Baselines are cached from earlier runs, so each tau takes minutes.
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
TAUS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
python zero_shot/run_zero_shot.py --target bio   --taus $TAUS --extra-datasets wmdp-cyber college_biology --tag sweep_bio_lower
python zero_shot/run_zero_shot.py --target cyber --taus $TAUS --tag sweep_cyber
python zero_shot/run_zero_shot.py --target bio --features-key bio_capital --taus $TAUS --extra-datasets wmdp-cyber college_biology --tag sweep_bio_capital

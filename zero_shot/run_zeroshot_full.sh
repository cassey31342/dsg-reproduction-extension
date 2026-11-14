#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=zs_full
#SBATCH --output=%x_%j.out
# Full run. ADJUST --time after the smoke run's timing. Usage: sbatch --export=ALL,TARGET=bio run_zeroshot_full.sh
#   TARGET=bio | cyber. Extra args via ARGS, e.g. ARGS="--extra-datasets wmdp-cyber college_biology"
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dsg
source ~/sae_guard/offline_env.sh
cd ~/sae_guard
python zero_shot/run_zero_shot.py --target "${TARGET:?set TARGET=bio or cyber}" ${ARGS}

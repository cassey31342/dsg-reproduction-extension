#!/bin/bash
# New env "dsg2" = Jeroen's setup (Python 3.10, requirements from his email, sae-lens 6.35.0).
module load Anaconda3
eval "$(conda shell.bash hook)"
conda env list | grep -q "^dsg2 " && echo "env dsg2 exists" || conda create -n dsg2 python=3.10 -y
conda activate dsg2
python --version
pip install -r ~/sae_guard/jeroen_env/requirements_jeroen.txt
python -m pip check || true
python -c "import torch,sae_lens,transformer_lens;from importlib.metadata import version as v;print('torch',torch.__version__,'sae-lens',v('sae-lens'),'TL',v('transformer-lens'))"
echo SETUP2_DONE

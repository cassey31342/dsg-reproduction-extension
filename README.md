# Dynamic SAE Guardrails: reproduction and extensions

Code and saved outputs for a study of Dynamic SAE Guardrails on WMDP-Bio and WMDP-Cyber, using Gemma-2-2B-IT. The extensions examine two-domain gating, threshold calibration, prompt transfer, and feature overlap.

Reference paper: https://arxiv.org/abs/2504.08192

## Contents

- `zero_shot/`: evaluation, extension, analysis, plotting, and Slurm scripts.
- `patches/`: compatibility patches for the separately supplied upstream implementation.
- `jeroen_env/`: an alternative environment specification used for compatibility investigation; not the working environment for all experiments.
- `Yiqi_neuronpedia/`: feature-search responses and comparison data.
- `results/`: saved experiment outputs; see its README for a map.
- `report/figures/`: figures used by plotting scripts. A final report is not included in this code-only preparation.

## Analyze saved outputs on CPU

From the repository root, with Python 3.11 or a compatible Python 3 version:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python zero_shot/calibrate_gates.py results/gate_stats_zs.npz --out /tmp/dsg_calibration.json
python zero_shot/analyze_transfer.py results/transfer_zs.npz --out /tmp/dsg_transfer.json
python zero_shot/make_plots.py
python zero_shot/plot_dual_gate.py
python zero_shot/make_report_figs.py
```

The CPU dependency list is minimal and unpinned, not a reconstruction of the original GPU environment. Plotting commands overwrite their generated figures/summary files. Original run directory names are retained because analysis scripts refer to them.

## GPU evaluation and upstream dependency

GPU evaluation requires the separately authorized upstream implementation, CUDA, model/data access, and compatible versions of PyTorch, TransformerLens and SAE Lens. The private upstream source, model weights, credentials and restricted corpora are excluded. This repository alone is sufficient for the documented saved-output analysis, but not a complete standalone GPU evaluation installation.

The study used SAE Lens 5.11.0 and TransformerLens 2.18.0 with the supplied compatibility patches. `jeroen_env/` documents an alternative compatibility investigation and should not be treated as the working setup. Obtain the upstream implementation from its maintainer and follow the patch paths against the matching revision.

The scripts import `evals.unlearning...`; expose the authorized upstream `src/dynamic_sae_guardrails` directory on PYTHONPATH. Some scripts additionally expect upstream data under `src/dynamic_sae_guardrails/evals/unlearning/data/` relative to the project root; use a local untracked layout matching this expectation for those tasks.

The supplied Slurm scripts preserve the original UBELIX assumptions: the `dsg` conda environment, `~/sae_guard` working directory, H100 GPU partition and a locally prepared `offline_env.sh`. These are site-specific launch scripts, not portable installation scripts. GPU runs also need baseline question-selection artifacts and authorized data/model caches; these are not fully bundled here.

Input configuration can be inspected without loading the GPU stack:

```bash
python zero_shot/run_zero_shot.py --target bio --dry-run
```

## Evaluation notes

Feature identifiers belong to a specific SAE and are not interchangeable between SAE configurations. Confidence intervals reflect question sampling, not variation across models or training seeds.

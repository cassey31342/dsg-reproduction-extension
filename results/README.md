# Saved experiment outputs

Original run names are preserved for the existing analysis scripts. JSON summaries contain configurations and aggregate scores; metrics files contain evaluation outputs. No model weights or restricted text corpora are included.

| Path/pattern | Experiment |
|---|---|
| `zeroshot_bio_*`, `zeroshot_cyber_*` | Single-domain evaluations, threshold scans and controls |
| `zeroshot_both_*` | Combined features and harmful-query feature experiments |
| `data_driven_table1/` | Data-driven Bio reproduction |
| `data_driven_cyber*` | Cyber reproduction and calibration-length diagnostics |
| `data_driven_bio_l059/`, `dd_l059/` | Same-SAE feature ranking and overlap |
| `dualgate_*` | Dual-gate evaluations and calibrated thresholds |
| `gate_stats_zs.npz`, `gate_calibration_zs.json` | Gate statistics and threshold calibration |
| `transfer_zs.npz`, `transfer_zs*.json` | Prompt transfer outputs and summaries |
| `exp2/` | Feature overlap/ablation planning and saved summaries |
| `dual_gate_ci.json` | Question-bootstrap intervals |

Plot map:

| Output | Script | Inputs |
|---|---|---|
| `sweep_zero_shot.png`, `cross_domain.png` | `zero_shot/make_plots.py` | Zero-shot run summaries |
| `dual_gate.png` | `zero_shot/plot_dual_gate.py` | Dual-gate and comparison run summaries |
| `report/figures/fire_rates.png` | `zero_shot/make_report_figs.py` | Gate statistics NPZ and calibration JSON |

Only load pickle result files from a trusted source. Some intermediate or diagnostic runs are retained because analysis scripts select runs by their original names; not every saved run is a final reported experiment.

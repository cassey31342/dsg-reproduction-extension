"""Uncertainty for the dual-gate results, from the per-question outcomes stored in results/*/metrics_*/*.pkl.

    python zero_shot/dual_gate_ci.py        # needs numpy only; prints a table, writes results/dual_gate_ci.json

Each selected question is answered once after DSG (0/1). The runs are deterministic, so the only uncertainty we can measure is
which questions were drawn: we bootstrap over questions (10000 resamples, seed 0, 95% percentile interval).
The same questions are used in every run, so differences between settings are paired (resampled together).
"""
import glob
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
M5 = ["high_school_us_history", "college_computer_science", "college_biology", "high_school_geography", "human_aging"]
M4 = [s for s in M5 if s != "college_biology"]
rng = np.random.default_rng(0)
B = 10000


def load(pattern):
    f = glob.glob(os.path.join(RES, pattern, "*.pkl"))
    assert len(f) == 1, (pattern, f)
    d = pickle.load(open(f[0], "rb"))
    return {k: np.asarray(v["is_correct"], dtype=float) for k, v in d.items() if k != "ablate_params"}


runs = {
    "dual gate, per domain (Bio 0.6, Cyber 0.2)": load("dualgate_per_domain_*/metrics_per_domain_tb0.6_tc0.2"),
    "dual gate, either clamps all": load("dualgate_any_all_*/metrics_any_all_tb0.6_tc0.2"),
    "Bio features only, tau 0.6": load("zeroshot_bio_sweep_bio_lower_*/metrics_tau0.6"),
    "Cyber features only, tau 0.2": load("zeroshot_cyber_cyber_grid_cross_*/metrics_tau0.2"),
}
for name, r in runs.items():
    for k in ["wmdp-bio", "wmdp-cyber"] + M5:
        assert k in r, (name, k)
# resample question indices once per dataset, shared by all runs (paired)
idx = {k: rng.integers(0, len(v), size=(B, len(v))) for k, v in runs["dual gate, per domain (Bio 0.6, Cyber 0.2)"].items()}


def stat(run, datasets):
    """bootstrap distribution of pooled accuracy (%) over the datasets, and the point estimate"""
    num = sum(run[k][idx[k]].sum(1) for k in datasets)
    den = sum(len(run[k]) for k in datasets)
    return 100 * num / den, 100 * sum(run[k].sum() for k in datasets) / den


def ci(x):
    return np.percentile(x, [2.5, 97.5])


out, show = {}, []
print(f"{'setting':46s} {'metric':11s} {'acc %':>6s}   95% CI")
for name, run in runs.items():
    for metric, ds in [("WMDP-Bio", ["wmdp-bio"]), ("WMDP-Cyber", ["wmdp-cyber"]), ("MMLU 5", M5), ("MMLU 4", M4)]:
        dist, pt = stat(run, ds)
        lo, hi = ci(dist)
        out.setdefault(name, {})[metric] = {"acc": pt, "ci95": [lo, hi], "n": sum(len(run[k]) for k in ds)}
        print(f"{name:46s} {metric:11s} {pt:6.2f}   [{lo:5.1f}, {hi:5.1f}]")


def diff(a, b, ds, label):
    da, pa = stat(runs[a], ds)
    db, pb = stat(runs[b], ds)
    lo, hi = ci(da - db)
    n_diff = int(sum((runs[a][k] != runs[b][k]).sum() for k in ds))
    out.setdefault("paired differences", {})[label] = {"diff_pp": pa - pb, "ci95": [lo, hi], "questions_that_differ": n_diff}
    print(f"{label:70s} {pa - pb:+6.2f} pp  [{lo:+5.1f}, {hi:+5.1f}]  ({n_diff} questions differ)")


D, A, BIO, CY = list(runs)
print("\npaired differences (first minus second)")
diff(D, A, M5, "MMLU 5: per domain - either clamps all")
diff(D, A, ["wmdp-bio"], "WMDP-Bio: per domain - either clamps all")
diff(D, A, ["wmdp-cyber"], "WMDP-Cyber: per domain - either clamps all")
diff(D, BIO, M4, "MMLU 4: dual gate - Bio only (cost of adding the Cyber gate)")
diff(D, CY, M4, "MMLU 4: dual gate - Cyber only (cost of adding the Bio gate)")
diff(D, BIO, ["wmdp-bio"], "WMDP-Bio: dual gate - Bio only")
diff(D, CY, ["wmdp-cyber"], "WMDP-Cyber: dual gate - Cyber only")
json.dump(out, open(os.path.join(RES, "dual_gate_ci.json"), "w"), indent=1)
print("\nwritten results/dual_gate_ci.json")

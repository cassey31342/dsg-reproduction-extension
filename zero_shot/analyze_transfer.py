"""Tables and intervals for the prompt-transfer experiment (transfer_*.npz from transfer.py). CPU, numpy only.

    python zero_shot/analyze_transfer.py results/transfer_zs.npz [--out results/transfer_zs_summary.json]

Per variant and threshold setting: accuracy of Bio, Cyber and MMLU (5 subjects pooled) with the guard and without it ('none'), and the
paired change caused by the guard with a 95% bootstrap interval over questions (10000 resamples, seed 0).
"""
import argparse
import json
import os

import numpy as np

MMLU5 = ["high_school_us_history", "college_computer_science", "college_biology", "high_school_geography", "human_aging"]
GROUPS = {"WMDP-Bio": ["wmdp-bio"], "WMDP-Cyber": ["wmdp-cyber"], "MMLU-5": MMLU5}
B = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.npz)
    acc = {}
    for k in z.files:
        _, variant, ds, cfg = k.split("|")
        acc.setdefault(variant, {}).setdefault(cfg, {})[ds] = z[k]
    rng = np.random.default_rng(0)
    out = {}
    for variant, by_cfg in acc.items():
        cfgs = [c for c in by_cfg if c != "none"]
        ds_all = [d for d in by_cfg["none"]]
        idx = {d: rng.integers(0, len(by_cfg["none"][d]), size=(B, len(by_cfg["none"][d]))) for d in ds_all}
        print(f"\n=== prompt variant: {variant} ===")
        print(f"{'setting':14s} " + " | ".join(f"{g:^38s}" for g in GROUPS))
        print(f"{'':14s} " + " | ".join(f"{'acc %':>8s} {'change vs no guard (pp) [95% CI]':>29s}" for _ in GROUPS))
        for cfg in ["none"] + cfgs:
            cells, rec = [], {}
            for g, dss in GROUPS.items():
                dss = [d for d in dss if d in by_cfg[cfg]]
                if not dss:
                    cells.append("n/a".center(38)); continue
                n = sum(len(by_cfg[cfg][d]) for d in dss)
                pt = float(100 * sum(by_cfg[cfg][d].sum() for d in dss) / n)
                if cfg == "none":
                    cells.append(f"{pt:8.2f} {'':>29s}"); rec[g] = {"acc": pt, "n": n}; continue
                dist = 100 * sum((by_cfg[cfg][d][idx[d]] - by_cfg["none"][d][idx[d]]).sum(1) for d in dss) / n
                ch = float(pt - 100 * sum(by_cfg["none"][d].sum() for d in dss) / n)
                lo, hi = map(float, np.percentile(dist, [2.5, 97.5]))
                cells.append(f"{pt:8.2f} {ch:+10.2f} [{lo:+6.1f}, {hi:+6.1f}]{'':>3s}")
                rec[g] = {"acc": pt, "change_pp": ch, "ci95": [lo, hi], "n": n}
            print(f"{cfg:14s} " + " | ".join(cells))
            out.setdefault(variant, {})[cfg] = rec
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("\nwritten", a.out)


if __name__ == "__main__":
    main()

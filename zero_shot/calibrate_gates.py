"""Calibrate the gate thresholds from gate_stats.py output (CPU, numpy only).

    python zero_shot/calibrate_gates.py results/gate_stats_full.npz [--alpha 0.05] [--out results/gate_calibration.json]

For gates i = 1..N with rho_i(x) from gate_stats.py:
  independent   tau_i = q-th percentile of rho_i on the calibration set (repo recipe: q = 95, WikiText test split, 1024-token chunks)
  bonferroni    tau_i = (1 - alpha/N)-th quantile of rho_i  (each gate gets an equal share of the false-positive budget)
  global        keep the independent tau_i and add one shift delta: gate i fires iff rho_i > tau_i + delta, with delta chosen so that
                P(any gate fires) <= alpha on the calibration set, i.e. delta = (1 - alpha)-quantile of max_i (rho_i - tau_i)
  paper         the paper's zero-shot values (Bio 0.6, Cyber 0.2), not calibrated by us
Held-out retain sets (never used to choose a threshold) report the false-positive rate of every gate and of "any gate".
Forget sets report how often the right gate fires (recall).  Intervals are 95% Wilson intervals.
"""
import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = {"bio": 0.6, "cyber": 0.2}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else float("nan"), "k": int(k), "n": int(n), "ci95": [lo, hi]}


def load(path):
    z = np.load(path, allow_pickle=False)
    groups = [str(g) for g in z["groups"]]
    sets = {k[4:]: z[k] for k in z.files if k.startswith("set|")}
    return groups, sets


def thresholds(groups, cal, alpha, q):
    """cal: [n, N] rho on the calibration set -> dict name -> {group: tau}"""
    N = len(groups)
    ind = {g: float(np.percentile(cal[:, j], q)) for j, g in enumerate(groups)}
    bonf = {g: float(np.quantile(cal[:, j], 1 - alpha / N)) for j, g in enumerate(groups)}
    s_max = np.max(cal - np.array([ind[g] for g in groups]), axis=1)
    delta = float(np.quantile(s_max, 1 - alpha))
    glob = {g: ind[g] + delta for g in groups}
    out = {"independent": ind, "bonferroni": bonf, "global": glob, "paper": {g: PAPER[g] for g in groups if g in PAPER}}
    return out, delta


def fires(rho, groups, tau):
    return np.stack([rho[:, j] > tau[g] for j, g in enumerate(groups)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=95.0, help="percentile of the independent calibration (repo: 95)")
    ap.add_argument("--cal-set", default="wiki_test_1024", help="set used to choose thresholds")
    ap.add_argument("--use-groups", nargs="+", default=None, help="only these gates (default: all groups in the file)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gate_calibration.json"))
    a = ap.parse_args()

    groups, sets = load(a.stats)
    if a.use_groups:
        cols = [groups.index(g) for g in a.use_groups]
        sets = {k: v[:, cols] for k, v in sets.items()}
        groups = list(a.use_groups)
    assert a.cal_set in sets, f"{a.cal_set} not in {sorted(sets)}"
    N = len(groups)
    result = {"stats_file": a.stats, "groups": groups, "alpha": a.alpha, "q": a.q, "cal_set": a.cal_set, "calibrations": {}}

    # calibration sources: the repo's (WikiText 1024 test split), shorter chunks, and format-matched MCQ prompts
    sources = [s for s in (a.cal_set, "wiki_test_128", "mcq_cal") if s in sets]
    print(f"gates: {groups}   alpha={a.alpha}   n per set: " + ", ".join(f"{s}={len(sets[s])}" for s in sources))
    all_tau = {}
    for src in sources:
        tau, delta = thresholds(groups, sets[src], a.alpha, a.q)
        all_tau[src] = tau
        result["calibrations"][src] = {"tau": tau, "delta": delta, "n": int(len(sets[src]))}
        print(f"\n[calibration set: {src}]  delta (global shift) = {delta:+.4f}")
        for name, t in tau.items():
            print(f"   {name:12s} " + "  ".join(f"{g}={t[g]:.4f}" for g in t))

    # held-out retain sets and forget sets: fire rates for every threshold rule of every calibration source
    retain_sets = [s for s in sets if s.startswith(("wiki_val", "mcq_cal", "mcq_eval", "mcq_bio_neighbours", "mcq_cyber_neighbours"))]
    pooled_eval = np.concatenate([sets[s] for s in sets if s.startswith("mcq_eval_")]) if any(s.startswith("mcq_eval_") for s in sets) else None
    held = dict((s, sets[s]) for s in retain_sets)
    if pooled_eval is not None:
        held["MMLU eval subjects (pooled)"] = pooled_eval
    forget_sets = [s for s in sets if s.startswith(("wmdp", "mixed", "bio_forget", "cyber_forget"))]

    rates = {}
    for src, taus in all_tau.items():
        rates[src] = {}
        for rule, tau in taus.items():
            rates[src][rule] = {}
            for s, rho in list(held.items()) + [(f, sets[f]) for f in forget_sets]:
                if s == src:
                    continue
                f = fires(rho, groups, tau)
                rates[src][rule][s] = {**{g: fmt_rate(f[:, j].sum(), len(f)) for j, g in enumerate(groups)},
                                       "any": fmt_rate(f.any(axis=1).sum(), len(f)), "all": fmt_rate(f.all(axis=1).sum(), len(f))}
    result["rates"] = rates

    # print the main table: calibration on the repo's set
    main_src = a.cal_set
    print(f"\n=== fire rates, thresholds calibrated on {main_src} (calibration set itself excluded) ===")
    show = [s for s in ["wiki_val_1024", "wiki_val_128", "mcq_cal", "MMLU eval subjects (pooled)", "wmdp-bio", "wmdp-cyber", "mixed_bio_cyber"] if s in held or s in sets]
    head = f"{'set':32s} {'n':>5s} | " + " | ".join(f"{r:^24s}" for r in ["paper", "independent", "global"])
    print(head)
    for s in show:
        row = f"{s:32s} {rates[main_src]['independent'].get(s, {}).get('any', {}).get('n', 0):5d} | "
        cells = []
        for rule in ("paper", "independent", "global"):
            r = rates[main_src].get(rule, {}).get(s)
            cells.append("n/a".center(24) if not r else "  ".join(f"{g[:3]} {100 * r[g]['rate']:5.1f}%" for g in groups) + f"  any {100 * r['any']['rate']:5.1f}%")
        print(row + " | ".join(cells))

    json.dump(result, open(a.out, "w"), indent=1)
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()

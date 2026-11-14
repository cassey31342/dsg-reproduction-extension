"""Figure for the report: how often does ANY gate fire on retain text, per input distribution, for two threshold rules.
Thresholds chosen on WikiText test, 1024-token chunks.  Wilson 95% intervals (WikiText-128: block bootstrap, 8 chunks per block).
    python zero_shot/make_report_figs.py        (CPU; reads results/gate_stats_zs.npz and results/gate_calibration_zs.json)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cal = json.load(open(os.path.join(ROOT, "results/gate_calibration_zs.json")))
z = np.load(os.path.join(ROOT, "results/gate_stats_zs.npz"))
groups = [str(g) for g in z["groups"]]
tau = cal["calibrations"]["wiki_test_1024"]["tau"]
sets = [("wiki_val_1024", "WikiText\n1024 tokens\n(held out)"), ("wiki_val_128", "WikiText\n128 tokens"),
        ("MMLU", "MMLU eval\nquestions"), ("mcq_cal", "10 other MMLU\nsubjects")]
rules = [("independent", "Independent (95th pct per gate)", "#9aa5b1"), ("global", "Global shift ($\\delta=+0.020$)", "#1f6feb")]

def rho_of(name):
    if name == "MMLU":
        return np.concatenate([z[k] for k in z.files if k.startswith("set|mcq_eval_")])
    return z["set|" + name]

def fire(rho, t):
    return np.stack([rho[:, j] > t[g] for j, g in enumerate(groups)], 1).any(1).astype(float)

def interval(f, name):
    n = len(f); p = f.mean()
    if name == "wiki_val_128":
        rng = np.random.default_rng(0); nb = n // 8; b = f[:nb * 8].reshape(nb, 8)
        bs = [b[rng.integers(0, nb, nb)].mean() for _ in range(5000)]
        return p, np.percentile(bs, [2.5, 97.5])
    zq = 1.96; d = 1 + zq * zq / n; c = (p + zq * zq / (2 * n)) / d
    h = zq * np.sqrt(p * (1 - p) / n + zq * zq / (4 * n * n)) / d
    return p, np.array([c - h, c + h])

fig, ax = plt.subplots(figsize=(6.4, 2.9))
w = 0.36
out = {}
for i, (rk, rl, col) in enumerate(rules):
    for j, (sk, sl) in enumerate(sets):
        p, (lo, hi) = interval(fire(rho_of(sk), tau[rk]), sk)
        out[f"{rk}|{sk}"] = [float(p), float(lo), float(hi)]
        x = j + (i - 0.5) * w
        ax.bar(x, 100 * p, w * 0.92, color=col, label=rl if j == 0 else None)
        ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="k", lw=0.9, capsize=2)
        ax.text(x, 100 * hi + 1.0, f"{100 * p:.1f}", ha="center", va="bottom", fontsize=7)
ax.axhline(5, color="#c0392b", lw=0.9, ls="--"); ax.text(3.62, 6.6, "5% target", color="#c0392b", fontsize=7, ha="right")
ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], fontsize=7.5)
ax.set_ylabel("Any gate fires on retain text (%)", fontsize=8); ax.set_ylim(0, 48)
ax.tick_params(axis="y", labelsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "report/figures/fire_rates.png"), dpi=220)
json.dump(out, open(os.path.join(ROOT, "results/fire_rates_fig.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

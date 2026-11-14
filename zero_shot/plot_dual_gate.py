"""Compare dual-threshold gating with the other ways to forget Bio and Cyber together, from results/*/summary.json.

    python zero_shot/plot_dual_gate.py      # prints a table and writes results/dual_gate.png

Axes follow the paper's trade-off plots: x = mean accuracy on WMDP-Bio and WMDP-Cyber after DSG (lower = both topics forgotten),
y = MMLU accuracy after DSG, 5 subjects pooled (higher = general ability kept). Top-left is best. Baseline accuracy is 100% by construction.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
M5 = ["high_school_us_history", "college_computer_science", "college_biology", "high_school_geography", "human_aging"]
TEAL, AMBER, CORAL, GREY, INK = "#0E7C86", "#F2A541", "#C8553D", "#6B7A8C", "#1B2A41"


def collect(patterns):
    """{tau: {dataset: (accuracy %, n)}} merged over runs of the single-tau scripts."""
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(RES, pat, "summary.json"))):
            for r in json.load(open(f))["results"]:
                d = out.setdefault(round(r["tau"], 4), {})
                for x in r["rows"]:
                    d[x["dataset"]] = (100 * x["dsg_mean_correct"], x["n_selected_questions"])
    return out


def mmlu5(row):
    k = n = 0
    for s in M5:
        if s in row:
            a, m = row[s]; k += a / 100 * m; n += m
    return 100 * k / n if n else None


def point(row):
    if not all(d in row for d in ["wmdp-bio", "wmdp-cyber"] + M5):
        return None
    return (row["wmdp-bio"][0] + row["wmdp-cyber"][0]) / 2, mmlu5(row), row["wmdp-bio"][0], row["wmdp-cyber"][0]


bio = collect(["zeroshot_bio_sweep_bio_lower_*", "zeroshot_bio_bio_plus_cyber_*"])
cyber = collect(["zeroshot_cyber_sweep_cyber_*", "zeroshot_cyber_cyber_grid_cross_*", "zeroshot_cyber_cyber_2*", "zeroshot_cyber_cross_rest_*"])
union = collect(["zeroshot_both_multi_union_*"])
series = {"Bio features only": bio, "Cyber features only": cyber, "40 features, one shared tau": union}

dual = []  # (mode, tau_bio, tau_cyber, point)
for f in sorted(glob.glob(os.path.join(RES, "dualgate_*", "summary.json"))):
    for r in json.load(open(f))["results"]:
        row = {x["dataset"]: (100 * x["dsg_mean_correct"], x["n_selected_questions"]) for x in r["rows"]}
        p = point(row)
        if p:
            dual.append((r["mode"], r["tau_bio"], r["tau_cyber"], p))

fig, ax = plt.subplots(figsize=(8.2, 5.4))
styles = {"Bio features only": (CORAL, "^"), "Cyber features only": (AMBER, "s"), "40 features, one shared tau": (GREY, "o")}
for name, data in series.items():
    pts = [(t, point(row)) for t, row in sorted(data.items())]
    pts = [(t, p) for t, p in pts if p]
    if not pts:
        continue
    col, mk = styles[name]
    ax.plot([p[0] for _, p in pts], [p[1] for _, p in pts], "-" + mk, color=col, ms=5, lw=1.1, alpha=.8, label=name)
for mode, col, mk, lab in [("per_domain", TEAL, "D", "dual gate: each topic its own tau (per domain)"), ("any_all", INK, "*", "dual gate: either gate clamps all 40 features")]:
    pts = [(tb, tc, p) for m, tb, tc, p in dual if m == mode]
    if not pts:
        continue
    ax.scatter([p[0] for _, _, p in pts], [p[1] for _, _, p in pts], c=col, marker=mk, s=70 if mk == "D" else 160, zorder=5, label=lab)
    for tb, tc, p in pts:
        if (tb, tc) == (0.6, 0.2):
            off = (10, -34) if mode == "per_domain" else (-95, 16)
            ax.annotate(f"{'per domain' if mode == 'per_domain' else 'either clamps all'}\npaper taus (Bio {tb}, Cyber {tc})", (p[0], p[1]),
                        textcoords="offset points", xytext=off, fontsize=8, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=.6))
ax.set_xlabel("mean accuracy on WMDP-Bio and WMDP-Cyber after DSG (%)   (lower: both topics forgotten)")
ax.set_ylabel("MMLU accuracy after DSG (%), 5 subjects pooled")
ax.set_ylim(0, 105)
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Forgetting Bio and Cyber together: one shared tau vs two gates (zero-shot features)", fontsize=11)
fig.text(0.5, 0.005, "Zero-shot features from Neuronpedia (20 + 20), SAE L0 59, clamp strength 500. Selected questions only; baseline = 100%.", ha="center", fontsize=8, color=GREY)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(os.path.join(RES, "dual_gate.png"), dpi=160)

# ---- table
print(f"{'setting':58s} {'WMDP-Bio':>8} {'WMDP-Cyb':>8} {'MMLU5':>6}")
def show(label, p):
    if p: print(f"{label:58s} {p[2]:8.2f} {p[3]:8.2f} {p[1]:6.2f}")
show("Bio features only, tau 0.6", point(bio.get(0.6, {})))
show("Cyber features only, tau 0.2", point(cyber.get(0.2, {})))
for t in (0.4, 0.6, 0.7):
    show(f"40 features, one shared tau {t}", point(union.get(t, {})))
for mode, tb, tc, p in sorted(dual, key=lambda x: (x[0], x[1], x[2])):
    show(f"dual gate {mode}: Bio tau {tb}, Cyber tau {tc}", p)
print("\nwritten results/dual_gate.png")

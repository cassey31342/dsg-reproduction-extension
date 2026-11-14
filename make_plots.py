"""Regenerate the tau-sweep figures from results/*/summary.json (Jeroen: make the plot re-generable, label axes, add legend).

    python zero_shot/make_plots.py            # writes results/sweep_zero_shot.png and results/cross_domain.png
Requires matplotlib. Paper points are approximate readings of Figure 6B (read by eye from the paper), marked as such.
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MMLU = {"bio": ["high_school_us_history", "college_computer_science", "high_school_geography", "human_aging"],
        "cyber": ["high_school_us_history", "college_biology", "high_school_geography", "human_aging"]}
# approximate readings of the paper's Figure 6B: (tau, WMDP accuracy %, MMLU relative accuracy %)
PAPER_BIO = [(.2, 24.5, 23.5), (.3, 24.8, 27.2), (.4, 26.0, 46.3), (.5, 27.5, 78.5), (.6, 31.0, 97.5), (.7, 48.0, 99.5)]
PAPER_CYBER = [(.05, 35.5, 52.0), (.1, 37.3, 70.5), (.15, 39.2, 93.3), (.2, 41.0, 98.7), (.3, 64.2, 100.0)]
TEAL, AMBER, CORAL, GREY = "#0E7C86", "#F2A541", "#C8553D", "#6B7A8C"


def collect(patterns):
    """-> {tau: {dataset: accuracy%}} merged over all matching runs (later runs overwrite equal taus)."""
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(RES, pat, "summary.json"))):
            for r in json.load(open(f))["results"]:
                d = out.setdefault(round(r["tau"], 4), {})
                for x in r["rows"]:
                    d[x["dataset"]] = (100 * x["dsg_mean_correct"], x["n_selected_questions"])
    return out


def mmlu_pooled(row, target):
    k = n = 0
    for s in MMLU[target]:
        if s in row:
            a, m = row[s]; k += a / 100 * m; n += m
    return 100 * k / n


bio_low = collect(["zeroshot_bio_sweep_bio_lower_*", "zeroshot_bio_bio_plus_cyber_*"])
bio_cap = collect(["zeroshot_bio_sweep_bio_capital_*"])
cyber = collect(["zeroshot_cyber_sweep_cyber_*", "zeroshot_cyber_cyber_grid_cross_*", "zeroshot_cyber_cyber_2*"])


def curve(data, wm, target):
    t = sorted(data)
    return [(x, data[x][wm][0], mmlu_pooled(data[x], target)) for x in t]


# ---------------------------------------------------------------- figure 1: Figure 6B style
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
for a, series, paper, title, xl in [
    (ax[0], [("ours: query 'biology' (= paper Table 16 IDs)", curve(bio_low, "wmdp-bio", "bio"), TEAL, "o"),
             ("ours: query 'Biology' (capital B, 11/20 in Table 16)", curve(bio_cap, "wmdp-bio", "bio"), AMBER, "^")],
     PAPER_BIO, "Zero-shot, Bio features (20 features)", "WMDP-Bio accuracy after DSG (%)"),
    (ax[1], [("ours: query 'Cybersecurity' (= paper Table 16 IDs)", curve(cyber, "wmdp-cyber", "cyber"), TEAL, "o")],
     PAPER_CYBER, "Zero-shot, Cyber features (20 features)", "WMDP-Cyber accuracy after DSG (%)")]:
    for lab, pts, col, mk in series:
        a.plot([p[1] for p in pts], [p[2] for p in pts], "-" + mk, color=col, label=lab, ms=6, lw=1.4)
        for p in pts:
            a.annotate(f"{p[0]:g}", (p[1], p[2]), textcoords="offset points", xytext=(5, 4 if col == TEAL else -10),
                       fontsize=8, color=col)
    a.plot([p[1] for p in paper], [p[2] for p in paper], "s", color=CORAL, ms=9, mfc="none", mew=1.8,
           label="paper Figure 6B (approximate reading)")
    a.set_title(title, fontsize=12)
    a.set_xlabel(xl)
    a.set_ylabel("MMLU accuracy after DSG (%), selected questions")
    a.set_ylim(0, 105)
    a.grid(alpha=.25)
    a.legend(fontsize=8, loc="lower right")
fig.text(0.5, 0.005, "Numbers next to the points are the threshold tau. Clamp strength 500, SAE layer 3 width 16k L0 59, one run (the pipeline is deterministic).",
         ha="center", fontsize=8, color=GREY)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(os.path.join(RES, "sweep_zero_shot.png"), dpi=160)

# ---------------------------------------------------------------- figure 2: cross-domain behaviour versus tau
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
specs = [(ax[0], bio_low, "Bio features applied to other datasets", 0.6,
          [("wmdp-bio", "WMDP-Bio (target)", CORAL, "o"), ("wmdp-cyber", "WMDP-Cyber", TEAL, "s"),
           ("college_biology", "MMLU college biology", AMBER, "^")]),
         (ax[1], cyber, "Cyber features applied to other datasets", 0.2,
          [("wmdp-cyber", "WMDP-Cyber (target)", CORAL, "o"), ("wmdp-bio", "WMDP-Bio", TEAL, "s"),
           ("college_biology", "MMLU college biology", AMBER, "^"),
           ("college_computer_science", "MMLU college computer science (only 9 questions)", GREY, "v")])]
for a, data, title, tau_paper, lines in specs:
    for key, lab, col, mk in lines:
        t = [x for x in sorted(data) if key in data[x]]
        a.plot(t, [data[x][key][0] for x in t], "-" + mk, color=col, label=lab, ms=5, lw=1.4)
    a.axvline(tau_paper, color="black", ls="--", lw=1)
    a.text(tau_paper, 3, f" paper tau = {tau_paper:g}", fontsize=8, va="bottom")
    a.set_title(title, fontsize=12)
    a.set_xlabel("threshold tau")
    a.set_ylabel("accuracy after DSG (%), selected questions")
    a.set_ylim(0, 105)
    a.grid(alpha=.25)
    a.legend(fontsize=8, loc="center right")
fig.text(0.5, 0.005, "Baseline accuracy is 100% by construction. At small tau the guardrail fires on almost every question, so every dataset drops.",
         ha="center", fontsize=8, color=GREY)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(os.path.join(RES, "cross_domain.png"), dpi=160)
print("written results/sweep_zero_shot.png and results/cross_domain.png")
print("cyber taus:", sorted(cyber))

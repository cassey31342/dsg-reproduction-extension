"""Thresholds for experiment 2 and the commands that evaluate them. CPU, numpy only.
    python zero_shot/exp2_plan.py results/gate_stats_dd.npz results/dd_l059/feature_lists.json results/exp2

Per k and per variant (full lists / shared features removed): thresholds from the repo's recipe (95th percentile of rho on WikiText test, 1024 tokens)
plus the global shift delta (joint false-positive rate <= 5% on that calibration set); and one shared gate over the union of both lists (95th percentile).
Writes ids_k<k>.json, plan.json and commands.sh (dual_gate.py runs).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_gates import load, thresholds  # noqa: E402

stats, lists, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
groups, sets = load(stats)
cal = sets["wiki_test_1024"]
col = {g: j for j, g in enumerate(groups)}
L = json.load(open(lists))["lists"]
plan, cmds = {}, []
for k in (20, 50, 100, 200):
    ids_path = os.path.join(out, f"ids_k{k}.json")
    json.dump({"bio": L["bio"][:k], "cyber": L["cyber"][:k]}, open(ids_path, "w"))
    shared = sorted(set(L["bio"][:k]) & set(L["cyber"][:k]))
    plan[k] = {"n_shared": len(shared), "shared": shared}
    for variant, suffix in (("full", ""), ("exclusive", "x")):
        if variant == "exclusive" and not shared:
            continue
        names = [f"bio_dd{k}{suffix}", f"cyber_dd{k}{suffix}"]
        sub = cal[:, [col[n] for n in names]]
        tau, delta = thresholds(["bio", "cyber"], sub, 0.05, 95)
        plan[k][variant] = {"independent": tau["independent"], "global": tau["global"], "delta": delta}
        flag = " --exclusive" if variant == "exclusive" else ""
        for rule in ("independent", "global"):
            t = tau[rule]
            cmds.append(f"python zero_shot/dual_gate.py --ids-json {ids_path}{flag} --modes per_domain --pairs {min(t['bio'], .999):.4f},{min(t['cyber'], .999):.4f} --tag exp2_k{k}_{variant}_{rule}")
    # one shared gate over all features (the shared-threshold baseline at this k)
    u = cal[:, [col[f"union_dd{k}"]]]
    tu = float(np.percentile(u[:, 0], 95))
    plan[k]["union"] = {"tau": tu}
    cmds.append(f"python zero_shot/dual_gate.py --ids-json {ids_path} --modes union --pairs {min(tu, .999):.4f},{min(tu, .999):.4f} --tag exp2_k{k}_union")
json.dump(plan, open(os.path.join(out, "plan.json"), "w"), indent=1)
open(os.path.join(out, "commands.sh"), "w").write("\n".join(cmds) + "\n")
print(json.dumps(plan, indent=1))
print(len(cmds), "commands")

"""Write the gate groups for experiment 2 (data-driven top-k features of both topics, SAE l0 59) as {"name": [ids]} for gate_stats.py.
    python zero_shot/exp2_groups.py results/dd_l059/feature_lists.json results/dd_l059/exp2_groups.json
Per k: bio_dd<k>, cyber_dd<k>, union_dd<k> (both lists together), and bio_dd<k>x / cyber_dd<k>x (features that appear in both lists removed).
"""
import json
import sys

L = json.load(open(sys.argv[1]))["lists"]
groups = {}
for k in (20, 50, 100, 200):
    b, c = L["bio"][:k], L["cyber"][:k]
    sh = set(b) & set(c)
    groups[f"bio_dd{k}"], groups[f"cyber_dd{k}"] = b, c
    groups[f"union_dd{k}"] = sorted(set(b) | set(c))
    groups[f"bio_dd{k}x"] = [i for i in b if i not in sh]
    groups[f"cyber_dd{k}x"] = [i for i in c if i not in sh]
json.dump(groups, open(sys.argv[2], "w"))
print({k: len(v) for k, v in groups.items()})

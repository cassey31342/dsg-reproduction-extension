"""Print threshold settings from a calibrate_gates.py JSON as 'name=tau_bio,tau_cyber' words (for transfer.py / dual_gate.py --pairs).
    python zero_shot/pairs_from_calibration.py results/gate_calibration.json [--names]
Settings: paper; ind_wiki / glob_wiki (calibrated on the repo's WikiText chunks); ind_mcq / glob_mcq (calibrated on retain MCQ prompts).
"""
import json
import sys

cal = json.load(open(sys.argv[1]))
src = {"wiki": cal["cal_set"], "mcq": "mcq_cal"}
out = ["paper=0.6,0.2"]
for short, s in src.items():
    c = cal["calibrations"].get(s)
    if not c:
        continue
    for rule, tag in (("independent", "ind"), ("global", "glob")):
        t = c["tau"][rule]
        out.append(f"{tag}_{short}={min(t['bio'], 0.999):.4f},{min(t['cyber'], 0.999):.4f}")
if "--pairs-only" in sys.argv:
    print(" ".join(o.split("=")[1] for o in out))
elif "--names" in sys.argv:
    print(" ".join(o.split("=")[0] for o in out))
else:
    print(" ".join(out))

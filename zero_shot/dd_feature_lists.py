"""Ranked data-driven feature lists (top 200) for Bio and Cyber on the SAME SAE, and how much they overlap (experiment 2). CPU, numpy only.

Re-implements the repo's selection (feature_activation.get_top_features_percentile) on the saved sparsity files:
    score = activation^2; keep features with forget score >= 5th percentile, retain score <= max, forget/retain ratio >= p_ratio percentile;
    order by forget score, descending.   (Bio: p_ratio 95; Cyber: p_ratio 90, as in the repo's configs.)
It is checked against the feature ids the repo itself saved with its runs (bio: results/dd_l059/bio/metrics, cyber likewise).

    python zero_shot/dd_feature_lists.py [--out results/dd_l059/feature_lists.json]
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = [20, 50, 100, 200]


def select(dirpath, p_ratio):
    f = np.loadtxt(os.path.join(dirpath, "feature_sparsity_forget.txt"))
    r = np.loadtxt(os.path.join(dirpath, "feature_sparsity_retain.txt"))
    fs, rs = f ** 2, r ** 2
    ratio = fs / (rs + 1e-21)
    sel = np.where((fs >= np.percentile(fs, 5)) & (rs <= np.percentile(rs, 100)) & (ratio >= np.percentile(ratio, p_ratio)))[0]
    return [int(i) for i in sel[np.argsort(-fs[sel])]]


def saved_ids(dirpath):
    """feature ids the repo saved in its metrics files, by n"""
    out = {}
    for f in glob.glob(os.path.join(dirpath, "metrics", "*.pkl")):
        d = pickle.load(open(f, "rb"))
        ids = [int(i) for i in np.asarray(d["ablate_params"]["features_to_ablate"])]
        out[len(ids)] = ids
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio-dir", default=os.path.join(ROOT, "results/dd_l059/bio"))
    ap.add_argument("--cyber-dir", default=os.path.join(ROOT, "results/dd_l059/cyber"))
    ap.add_argument("--zero-shot", default=os.path.join(ROOT, "zero_shot/zero_shot_features.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/dd_l059/feature_lists.json"))
    a = ap.parse_args()

    lists = {"bio": select(a.bio_dir, 95), "cyber": select(a.cyber_dir, 90)}
    for name, d in (("bio", a.bio_dir), ("cyber", a.cyber_dir)):
        for n, ids in saved_ids(d).items():
            ok = lists[name][:n] == ids
            print(f"check {name}: repo-saved top-{n} == re-implemented top-{n}: {ok}")
            assert ok, f"re-implementation does not match the repo for {name} n={n}"
    zs = json.load(open(a.zero_shot))
    zero = {"bio": [int(i) for i in zs["bio"]["ids"]], "cyber": [int(i) for i in zs["cyber"]["ids"]]}

    result = {"lists": {k: v[:200] for k, v in lists.items()}, "n_selected": {k: len(v) for k, v in lists.items()}, "overlap": {}}
    print("\nBio vs Cyber, data-driven, top-k by forget score (same SAE, l0 59)")
    print(f"{'k':>4s} {'shared':>7s} {'Jaccard':>8s}  shared feature ids")
    for k in KS:
        b, c = lists["bio"][:k], lists["cyber"][:k]
        sh = sorted(set(b) & set(c))
        result["overlap"][str(k)] = {"shared": sh, "n_shared": len(sh), "jaccard": jaccard(b, c)}
        print(f"{k:4d} {len(sh):7d} {jaccard(b, c):8.3f}  {sh[:12]}{' ...' if len(sh) > 12 else ''}")
    print("\nzero-shot (Neuronpedia) vs data-driven, same SAE")
    result["zero_shot_vs_dd"] = {}
    for topic in ("bio", "cyber"):
        for k in KS:
            sh = sorted(set(zero[topic]) & set(lists[topic][:k]))
            result["zero_shot_vs_dd"][f"{topic}@{k}"] = {"n_shared": len(sh), "shared": sh}
            print(f"  {topic} zero-shot (20) vs data-driven top-{k}: {len(sh)} shared")
    print("  Bio zero-shot vs Cyber data-driven top-200:", len(set(zero["bio"]) & set(lists["cyber"][:200])),
          "   Cyber zero-shot vs Bio data-driven top-200:", len(set(zero["cyber"]) & set(lists["bio"][:200])))
    json.dump(result, open(a.out, "w"), indent=1)
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()

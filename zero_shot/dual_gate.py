"""Dual-threshold gating (extension recommended by Jeroen, meeting of 27 October).

Problem it solves: with the zero-shot Bio and Cyber features put together (40 features), ONE shared threshold tau cannot
forget both topics and keep MMLU (see results/zeroshot_both_multi_union_*). Here each topic keeps its OWN gate and its own
pre-calibrated threshold (paper values: Bio 0.6, Cyber 0.2), and we intervene whenever either gate is triggered.

Two modes (both clamp to -multiplier, like the repo's anthropic_clamp_resid_SAE_features):
  per_domain  each topic's features are clamped only when that topic's own gate fires (two independent guardrails)
  any_all     if either gate fires, the features of BOTH topics are clamped (literal reading: "either triggers")

The hook is our own code. The repo's evaluation code (evals.unlearning.utils.metrics.calculate_metrics_list) is used unchanged;
we only replace the clamp function it calls, in this process (nothing in the repo is edited).
For one topic alone, the hook gives exactly the repo's result (checked in test_dual_gate.py).

Examples:
    python zero_shot/dual_gate.py --dry-run
    python zero_shot/dual_gate.py --tau-bio 0.5 0.6 0.7 --tau-cyber 0.15 0.2 0.25 0.3
    python zero_shot/dual_gate.py --modes any_all --tau-bio 0.6 --tau-cyber 0.2 --tag any_all
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_zero_shot as Z  # noqa: E402  (only stdlib at import time)


def make_dual_gate_hook():
    """Return the hook function. Imports torch lazily so that --dry-run works anywhere."""
    import torch

    def dual_gate_hook(resid, hook, sae, features_to_ablate, multiplier, activation_threshold, random=False):
        # activation_threshold carries the whole gate configuration: {"mode": ..., "groups": [(ids, tau), ...]}
        cfg = activation_threshold
        acts = sae.encode(resid)
        acts[:, 0, :] = 0.0                       # as in the repo: the BOS token is never clamped
        err = resid - sae.decode(acts)            # reconstruction error is added back at the end

        groups = []
        for ids, tau in cfg["groups"]:
            tgt = acts[:, :, ids]
            pos = (tgt > 0).sum(dim=2) > 0        # [batch, seq]: some feature of this topic is active at this token
            rate = pos.sum(dim=1) / pos.shape[1]  # [batch]: share of tokens where it is active
            groups.append((ids, pos, rate > tau))  # gate fires for a prompt if the share is above this topic's tau

        if cfg["mode"] == "per_domain":
            plan = [(ids, pos & gate.unsqueeze(1)) for ids, pos, gate in groups]
        elif cfg["mode"] == "any_all":
            gate_any = torch.stack([g for _, _, g in groups]).any(dim=0)
            pos_any = torch.stack([p for _, p, _ in groups]).any(dim=0)
            all_ids = [i for ids, _, _ in groups for i in ids]
            plan = [(all_ids, pos_any & gate_any.unsqueeze(1))]
        else:
            raise ValueError(cfg["mode"])

        for ids, mask in plan:                    # masks were computed before any clamping
            tgt = acts[:, :, ids]
            acts[:, :, ids] = torch.where(mask.unsqueeze(2), torch.full_like(tgt, -multiplier), tgt)
        return sae.decode(acts) + err

    return dual_gate_hook


def parse_args():
    p = argparse.ArgumentParser(description="Dual-threshold gating, zero-shot Bio + Cyber features")
    p.add_argument("--features-file", default=os.path.join(HERE, "zero_shot_features.json"))
    p.add_argument("--bio-key", default="bio")
    p.add_argument("--cyber-key", default="cyber")
    p.add_argument("--tau-bio", type=float, nargs="+", default=[0.6], help="Bio gate thresholds (paper zero-shot: 0.6)")
    p.add_argument("--tau-cyber", type=float, nargs="+", default=[0.2], help="Cyber gate thresholds (paper zero-shot: 0.2)")
    p.add_argument("--pairs", nargs="+", default=None, help='threshold pairs "tau_bio,tau_cyber" (instead of the grid --tau-bio x --tau-cyber)')
    p.add_argument("--ids-json", default=None, help='feature ids of both topics: {"bio": [...], "cyber": [...]} (instead of the features file)')
    p.add_argument("--exclusive", action="store_true", help="drop features that appear in both topics' lists (Exclusive Only)")
    p.add_argument("--modes", nargs="+", default=["per_domain"], choices=["per_domain", "any_all", "union"], help="union = ONE gate over all features with the Bio threshold (the shared-threshold baseline)")
    p.add_argument("--multiplier", type=int, default=500)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--sae-id", default=Z.SAE_ID)
    p.add_argument("--mcq-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Z.DEFAULTS["both"]
    if args.ids_json:
        d = json.load(open(args.ids_json))
        bio_ids, cy_ids = [int(x) for x in d["bio"]], [int(x) for x in d["cyber"]]
    else:
        bio_ids, bio_entry = Z.load_features(args.features_file, args.bio_key)
        cy_ids, cy_entry = Z.load_features(args.features_file, args.cyber_key)
    shared = sorted(set(bio_ids) & set(cy_ids))
    if args.exclusive:
        bio_ids, cy_ids = [i for i in bio_ids if i not in shared], [i for i in cy_ids if i not in shared]
    else:
        assert not shared or args.ids_json, "Bio and Cyber feature sets must be disjoint (or pass --ids-json)"
    datasets = args.datasets or ([cfg["wmdp"]] + cfg.get("extra_wmdp", []) + cfg["mmlu"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"dualgate{'_' + args.tag if args.tag else ''}_{stamp}"
    run_dir = os.path.join(args.results_dir, run_name)
    artifacts_folder = os.path.join(args.results_dir, "artifacts", Z.MODEL_NAME)
    pairs = [tuple(float(x) for x in q.split(",")) for q in args.pairs] if args.pairs else [(tb, tc) for tb in args.tau_bio for tc in args.tau_cyber]
    combos = [(m, tb, tc) for m in args.modes for tb, tc in pairs]
    plan = {"run": run_name, "bio_key": args.bio_key, "cyber_key": args.cyber_key, "n_bio": len(bio_ids), "n_cyber": len(cy_ids), "n_shared": len(shared), "exclusive": args.exclusive,
            "sae": f"{Z.SAE_RELEASE}/{args.sae_id}", "multiplier": args.multiplier, "combos": combos, "datasets": datasets,
            "seed": args.seed, "run_dir": run_dir}
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        print("DRY RUN OK (nothing loaded)")
        return

    import numpy as np
    import torch
    assert torch.cuda.is_available(), "CUDA required: the repo hard-codes 'cuda' in metrics.py"
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    import evals.unlearning.utils.metrics as M

    M.anthropic_clamp_resid_SAE_features = make_dual_gate_hook()   # metrics.modify_model looks this name up when it builds the hook
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)

    model = HookedTransformer.from_pretrained_no_processing(Z.MODEL_NAME, device="cuda", dtype=torch.float32)
    sae, _, _ = SAE.from_pretrained(release=Z.SAE_RELEASE, sae_id=args.sae_id, device="cuda")
    sae = sae.to(device="cuda", dtype=torch.float32)
    all_ids = sorted(set(bio_ids) | set(cy_ids))

    results = []
    for mode, tb, tc in combos:
        t1 = time.time()
        gate_cfg = ({"mode": "per_domain", "groups": [(all_ids, tb)]} if mode == "union"
                    else {"mode": mode, "groups": [(bio_ids, tb), (cy_ids, tc)]})
        sweep = {"features_to_ablate": [np.array(all_ids)], "multiplier": [args.multiplier],
                 "activation_threshold": {str(len(all_ids)): gate_cfg}}
        label = f"{mode}_tb{tb}_tc{tc}"
        metrics_list = M.calculate_metrics_list(
            model, args.mcq_batch_size, sae, {"intervention_method": "clamp_feature_activation"}, sweep,
            artifacts_folder, args.force, dataset_names=datasets, target_metric="correct",
            save_metrics=True, save_metrics_dir=os.path.join(run_dir, f"metrics_{label}"),
            retain_threshold=f"dual-{label}", seed=args.seed)
        baseline, dsg = metrics_list[0], metrics_list[1]
        rows, agg = Z.summarize(datasets, "both", baseline, dsg, artifacts_folder)
        r = {"mode": mode, "tau_bio": tb, "tau_cyber": tc, "rows": rows, "aggregate": agg, "elapsed_s": time.time() - t1}
        results.append(r)
        by = {x["dataset"]: 100 * x["dsg_mean_correct"] for x in rows}
        mm = agg["mmlu_side_effect_accuracy_after_dsg"]
        print(f"\n=== {mode}  tau_bio={tb}  tau_cyber={tc}:  WMDP-Bio {by['wmdp-bio']:.2f}  WMDP-Cyber {by['wmdp-cyber']:.2f}  "
              f"MMLU(5 subjects) {100 * mm:.2f}   ({r['elapsed_s'] / 60:.1f} min)")

    from importlib.metadata import version
    env = {"torch": torch.__version__, "sae_lens": version("sae-lens"), "transformer_lens": version("transformer-lens"),
           "gpu": torch.cuda.get_device_name(0), "argv": sys.argv}
    json.dump({"plan": plan, "env": env, "results": results,
               "note": "Own hook (zero_shot/dual_gate.py); repo evaluation code unchanged. Scores on the selected questions."},
              open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print("summary written to", os.path.join(run_dir, "summary.json"))


if __name__ == "__main__":
    main()

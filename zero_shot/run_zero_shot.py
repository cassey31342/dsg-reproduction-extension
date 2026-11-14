"""Zero-shot DSG evaluation (paper Sec. 4.4, App. J Table 14, App. M).

No forget/retain data and no activation collection: the 20 feature IDs come from
Neuronpedia (zero_shot_features.json) and the dynamic threshold tau is a fixed value
(paper: 0.6 for WMDP-Bio, 0.2 for WMDP-Cyber), not a percentile from retain data.

It only calls the private repo's existing evaluation code
(evals.unlearning.utils.metrics.calculate_metrics_list); nothing in the repo is changed.
Run on a CUDA machine (the repo hard-codes "cuda" in metrics.py). Use --dry-run anywhere.

Example (bio features, paper setting):
    python zero_shot/run_zero_shot.py --target bio
Quick timing run on one small dataset:
    python zero_shot/run_zero_shot.py --target bio --datasets college_computer_science --tag smoke
Extension (bio features, also test Cyber questions):
    python zero_shot/run_zero_shot.py --target bio --extra-datasets wmdp-cyber college_biology
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Paper (App. J, Table 14) settings
DEFAULTS = {
    "bio": {"taus": [0.6], "wmdp": "wmdp-bio",
            "mmlu": ["high_school_us_history", "college_computer_science",
                     "high_school_geography", "human_aging"]},      # = UnlearningEvalConfig.dataset_names
    "cyber": {"taus": [0.2], "wmdp": "wmdp-cyber",
              "mmlu": ["high_school_us_history", "college_biology",
                       "high_school_geography", "human_aging"]},    # = UnlearningEvalConfigCyber.dataset_names
    # Multi-task (Figure 4 style): Bio and Cyber features clamped together (40 features, one shared tau).
    # tau has no paper value here: sweep it. MMLU = union of both configs' subjects.
    "both": {"taus": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "wmdp": "wmdp-bio", "extra_wmdp": ["wmdp-cyber"],
             "mmlu": ["high_school_us_history", "college_computer_science", "college_biology",
                      "high_school_geography", "human_aging"]},
}
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_3/width_16k/average_l0_59"   # paper App. J: zero-shot uses l0 59 (data-driven DSG uses l0 142)
MODEL_NAME = "gemma-2-2b-it"


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot DSG evaluation")
    p.add_argument("--target", choices=["bio", "cyber", "both"], required=True,
                   help="Which Neuronpedia feature set to clamp (and which paper setting to use)")
    p.add_argument("--features-file", default=os.path.join(HERE, "zero_shot_features.json"))
    p.add_argument("--features-key", default=None,
                   help="Key in the features file (default: same as --target). e.g. bio_capital = Neuronpedia 'Biology'")
    p.add_argument("--taus", type=float, nargs="+", default=None,
                   help="Static thresholds tau. Default: paper value (bio 0.6, cyber 0.2). Paper sweep: 0.1..0.9")
    p.add_argument("--multiplier", type=int, default=500, help="Clamp strength c (paper: 500)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Override the evaluated datasets (default: the target's WMDP set + its 4 MMLU subjects)")
    p.add_argument("--extra-datasets", nargs="*", default=[],
                   help="Datasets added on top of the defaults (e.g. wmdp-cyber for the cross-domain extension)")
    p.add_argument("--sae-id", default=SAE_ID)
    p.add_argument("--mcq-batch-size", type=int, default=1, help="Keep 1 (repo default; tau is applied per prompt)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="Label added to the run folder name")
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--force", action="store_true", help="Recompute metrics even if cached files exist")
    p.add_argument("--dry-run", action="store_true", help="Check inputs and print the plan; load nothing heavy")
    return p.parse_args()


def load_features(path, target):
    d = json.load(open(path))
    entry = d[target]
    ids = [int(x) for x in entry["ids"]]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    return ids, entry


def summarize(datasets, target, baseline, dsg, artifacts_folder):
    rows, mmlu_correct, mmlu_total = [], 0, 0
    mmlu_set = set(DEFAULTS[target]["mmlu"])
    for ds in datasets:
        name = ds if ds.startswith("wmdp") else "mmlu-" + ds.replace("_", "-")
        ids_file = os.path.join(artifacts_folder, "data/question_ids/all", f"{name}_correct.csv")
        n_selected = None
        if os.path.exists(ids_file):
            with open(ids_file) as f:
                n_selected = sum(1 for line in f if line.strip())
        b, a = baseline[ds], dsg[ds]
        n = len(a["is_correct"])
        rows.append({"dataset": ds, "n_selected_questions": n_selected, "n_is_correct_entries": n,
                     "baseline_mean_correct": float(b["mean_correct"]),
                     "dsg_mean_correct": float(a["mean_correct"]),
                     "change_pp": 100.0 * (float(a["mean_correct"]) - float(b["mean_correct"]))})
        if ds in mmlu_set:
            mmlu_correct += float(a["total_correct"]) if "total_correct" in a else float(a["mean_correct"]) * n
            mmlu_total += n
    agg = {"mmlu_side_effect_accuracy_after_dsg": (mmlu_correct / mmlu_total) if mmlu_total else None,
           "mmlu_subjects_used": sorted(mmlu_set & set(datasets))}
    return rows, agg


def main():
    args = parse_args()
    cfg = DEFAULTS[args.target]
    key = args.features_key or args.target
    ids, entry = load_features(args.features_file, key)
    taus = args.taus or cfg["taus"]
    datasets = args.datasets or ([cfg["wmdp"]] + cfg.get("extra_wmdp", []) + cfg["mmlu"])
    datasets = datasets + [d for d in args.extra_datasets if d not in datasets]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"zeroshot_{args.target}{'_' + args.tag if args.tag else ''}_{stamp}"
    run_dir = os.path.join(args.results_dir, run_name)
    # Baseline results and the "answered correctly under all 24 permutations" question lists are cached here and
    # shared between runs (they do not depend on the SAE or features), so baselines are only computed once.
    artifacts_folder = os.path.join(args.results_dir, "artifacts", MODEL_NAME)

    plan = {"run": run_name, "target": args.target, "feature_key": key, "feature_query": entry["query"],
            "feature_query_date": entry["query_date"], "n_features": len(ids), "feature_ids": ids,
            "sae": f"{SAE_RELEASE}/{args.sae_id}", "multiplier": args.multiplier, "taus": taus,
            "datasets": datasets, "mcq_batch_size": args.mcq_batch_size, "seed": args.seed,
            "artifacts_folder": artifacts_folder, "run_dir": run_dir}
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        print("DRY RUN OK (nothing loaded)")
        return

    import numpy as np
    import torch
    assert torch.cuda.is_available(), "CUDA required: the repo hard-codes 'cuda' in metrics.py"
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from evals.unlearning.utils.metrics import calculate_metrics_list

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)

    t0 = time.time()
    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=device, dtype=torch.float32)
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=args.sae_id, device=device)
    sae = sae.to(device=device, dtype=torch.float32)
    print(f"loaded model + SAE in {time.time() - t0:.1f}s; hook={sae.cfg.hook_name}, d_sae={sae.cfg.d_sae}")

    from importlib.metadata import version
    env = {"torch": torch.__version__, "sae_lens": version("sae-lens"),
           "transformer_lens": version("transformer-lens"), "gpu": torch.cuda.get_device_name(0),
           "argv": sys.argv}

    all_results = []
    for tau in taus:
        t1 = time.time()
        sweep = {"features_to_ablate": [np.array(ids)],
                 "multiplier": [args.multiplier],
                 "activation_threshold": {str(len(ids)): tau}}
        metrics_dir = os.path.join(run_dir, f"metrics_tau{tau}")
        metrics_list = calculate_metrics_list(
            model, args.mcq_batch_size, sae,
            {"intervention_method": "clamp_feature_activation"}, sweep,
            artifacts_folder, args.force,
            dataset_names=datasets, target_metric="correct",
            save_metrics=True, save_metrics_dir=metrics_dir,
            retain_threshold=f"zs-tau{tau}", seed=args.seed)
        baseline, dsg = metrics_list[0], metrics_list[1]
        rows, agg = summarize(datasets, args.target, baseline, dsg, artifacts_folder)
        elapsed = time.time() - t1
        result = {"tau": tau, "rows": rows, "aggregate": agg, "elapsed_s": elapsed}
        all_results.append(result)

        print(f"\n=== {args.target} zero-shot, tau={tau}, c={args.multiplier}  ({elapsed / 60:.1f} min) ===")
        print(f"{'dataset':28s} {'n_sel':>6s} {'baseline':>9s} {'DSG':>8s} {'change(pp)':>11s}")
        for r in rows:
            print(f"{r['dataset']:28s} {str(r['n_selected_questions']):>6s} "
                  f"{100 * r['baseline_mean_correct']:9.2f} {100 * r['dsg_mean_correct']:8.2f} {r['change_pp']:11.2f}")
        if agg["mmlu_side_effect_accuracy_after_dsg"] is not None:
            print(f"MMLU side-effect accuracy after DSG: {100 * agg['mmlu_side_effect_accuracy_after_dsg']:.2f}%")

    summary = {"plan": plan, "env": env, "results": all_results,
               "note": "Scores are measured on the question subset the baseline answers correctly under all 24 "
                       "permutations (paper Sec. 4.1). Not the paper's numbers."}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsummary written to {os.path.join(run_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

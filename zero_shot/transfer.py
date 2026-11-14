"""Accuracy with and without the dual gate when the prompt changes (experiment 3, prompt transfer). GPU.

Features and thresholds are FIXED (chosen on the original prompts); the questions are the same selected questions, but the prompt is
changed (see prompt_variants.py). For each (variant, dataset, setting) the 0/1 outcome of every question is saved, so intervals and
paired differences are computed offline.  'none' = model without guardrail (the baseline is no longer 100% once the prompt changes).

    python zero_shot/transfer.py --dry-run
    python zero_shot/transfer.py --configs paper=0.6,0.2 independent=0.30,0.10 --tag first
    (a threshold pair per setting: name=tau_bio,tau_cyber; mode per domain; zero-shot Bio and Cyber features unless --ids-json)
"""
import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import prompt_variants as PV  # noqa: E402
import run_zero_shot as Z  # noqa: E402
import dual_gate as DG  # noqa: E402

MMLU5 = ["high_school_us_history", "college_computer_science", "college_biology", "high_school_geography", "human_aging"]


def parse_args():
    p = argparse.ArgumentParser(description="Prompt transfer of the dual gate")
    p.add_argument("--configs", nargs="+", required=True, help="name=tau_bio,tau_cyber")
    p.add_argument("--variants", nargs="+", default=PV.VARIANTS)
    p.add_argument("--features-file", default=os.path.join(HERE, "zero_shot_features.json"))
    p.add_argument("--ids-json", default=None)
    p.add_argument("--multiplier", type=int, default=500)
    p.add_argument("--sae-id", default=Z.SAE_ID)
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ids_json:
        d = json.load(open(args.ids_json))
        bio_ids, cy_ids = d["bio"], d["cyber"]
    else:
        bio_ids = Z.load_features(args.features_file, "bio")[0]
        cy_ids = Z.load_features(args.features_file, "cyber")[0]
    cfgs = {}
    for c in args.configs:
        name, v = c.split("=")
        tb, tc = (float(x) for x in v.split(","))
        cfgs[name] = (tb, tc)
    datasets = ["wmdp-bio", "wmdp-cyber"] + MMLU5
    out_path = os.path.join(args.results_dir, f"transfer{'_' + args.tag if args.tag else ''}.npz")
    print(json.dumps({"configs": cfgs, "variants": args.variants, "datasets": datasets, "n_bio": len(bio_ids), "n_cyber": len(cy_ids), "out": out_path}, indent=1))
    if args.dry_run:
        print("DRY RUN OK (nothing loaded)")
        return

    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    import evals.unlearning.utils.metrics as M
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)
    artifacts = os.path.join(args.results_dir, "artifacts", Z.MODEL_NAME)
    model = HookedTransformer.from_pretrained_no_processing(Z.MODEL_NAME, device="cuda", dtype=torch.float32)
    sae = SAE.from_pretrained(release=Z.SAE_RELEASE, sae_id=args.sae_id, device="cuda")[0].to(device="cuda", dtype=torch.float32)
    hook = DG.make_dual_gate_hook()
    all_ids = sorted(set(bio_ids) | set(cy_ids))

    def accuracy(prompts, answers):
        probs = M.get_output_probs_abcd(model, prompts, batch_size=1, n_batches=len(prompts), verbose=False)
        return (probs.argmax(dim=1).cpu().numpy() == np.asarray(answers)).astype(np.float32)

    out, t0 = {}, time.time()
    for variant in args.variants:
        for ds in datasets:
            if variant == "swap_preamble" and not ds.startswith("wmdp"):
                continue
            name = ds if ds.startswith("wmdp") else "mmlu-" + ds.replace("_", "-")
            subset = np.genfromtxt(os.path.join(artifacts, "data/question_ids/all", f"{name}_correct.csv"), ndmin=1, dtype=int)
            prompts, answers = PV.build(ds, subset, variant)
            model.reset_hooks()
            out[f"acc|{variant}|{ds}|none"] = accuracy(prompts, answers)
            line = f"{variant:14s} {ds:26s} n={len(prompts):4d}  none {100 * out[f'acc|{variant}|{ds}|none'].mean():6.2f}"
            for cname, (tb, tc) in cfgs.items():
                gate = {"mode": "per_domain", "groups": [(bio_ids, tb), (cy_ids, tc)]}
                model.reset_hooks()
                model.add_hook(sae.cfg.hook_name, partial(hook, sae=sae, features_to_ablate=all_ids, multiplier=args.multiplier, activation_threshold=gate))
                out[f"acc|{variant}|{ds}|{cname}"] = accuracy(prompts, answers)
                line += f"  {cname} {100 * out[f'acc|{variant}|{ds}|{cname}'].mean():6.2f}"
            model.reset_hooks()
            print(line, f"({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(out_path, **out)
    json.dump({"configs": cfgs, "variants": args.variants, "n_bio": len(bio_ids), "n_cyber": len(cy_ids), "argv": sys.argv},
              open(out_path.replace(".npz", ".json"), "w"))
    print("written", out_path)


if __name__ == "__main__":
    main()

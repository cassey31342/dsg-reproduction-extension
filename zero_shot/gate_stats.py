"""Per-prompt gate statistics rho_i(x) for several feature groups, on many prompt sets. GPU, but only runs the model up to layer 3.

rho_i(x) = share of tokens of x at which at least one feature of group i is active (BOS never counts), exactly as in the repo's
clamp function. Everything else (thresholds, false-positive rates, recall, how the gates of two topics co-fire) is computed offline
from the saved numbers by calibrate_gates.py, so no model needs to be loaded again.

    python zero_shot/gate_stats.py --dry-run
    python zero_shot/gate_stats.py --tag first                 # zero-shot Bio and Cyber groups, all prompt sets
    python zero_shot/gate_stats.py --groups-json groups.json   # extra groups: {"name": [feature ids]}

Prompt sets (ids in the output file):
  wiki_test_1024 / wiki_val_1024   WikiText-2 test / validation split, 1024-token chunks (the repo calibrates on the test split)
  wiki_test_128  / wiki_val_128    same text, 128-token chunks
  mcq_cal                          MMLU questions of subjects NOT used in any evaluation, in the repo's evaluation prompt format
  mcq_eval_<subject>               the MMLU questions the evaluation uses (selected question ids)
  mcq_bio_neighbours / mcq_cyber_neighbours   MMLU subjects close to Bio / Cyber (all questions)
  wmdp-bio, wmdp-cyber             selected WMDP questions (what the evaluation uses)
  wmdp-bio|<variant>, wmdp-cyber|<variant>    the same questions in changed prompts (see VARIANTS)
  mixed_bio_cyber                  a Bio question and a Cyber question in the same prompt
  bio_forget_corpus_1024 / _128, cyber_forget_corpus_1024 / _128   forget-corpus text chunks (sampled)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import prompt_variants as PV  # noqa: E402
import run_zero_shot as Z  # noqa: E402  (stdlib only at import time)

CAL_SUBJECTS = ["philosophy", "high_school_world_history", "high_school_physics", "high_school_macroeconomics", "jurisprudence",
                "marketing", "sociology", "high_school_mathematics", "prehistory", "world_religions"]
BIO_NEIGHBOURS = ["high_school_biology", "virology", "anatomy", "medical_genetics", "college_medicine", "nutrition"]
CYBER_NEIGHBOURS = ["computer_security", "high_school_computer_science", "machine_learning", "electrical_engineering"]
EVAL_MMLU = ["high_school_us_history", "college_computer_science", "college_biology", "high_school_geography", "human_aging"]
VARIANTS = PV.VARIANTS[1:]   # original = the repo's evaluation prompt (see prompt_variants.py)


def parse_args():
    p = argparse.ArgumentParser(description="Gate statistics rho_i(x)")
    p.add_argument("--features-file", default=os.path.join(HERE, "zero_shot_features.json"))
    p.add_argument("--feature-keys", nargs="+", default=["bio", "cyber"], help="groups taken from the features file")
    p.add_argument("--groups-json", default=None, help='extra groups: {"name": [ids]}')
    p.add_argument("--sets", nargs="+", default=None, help="only these prompt sets (default: all)")
    p.add_argument("--sae-id", default=Z.SAE_ID)
    p.add_argument("--n-corpus", type=int, default=300, help="forget-corpus chunks per length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def selected_ids(artifacts, dataset):
    name = dataset if dataset.startswith("wmdp") else "mmlu-" + dataset.replace("_", "-")
    return np.genfromtxt(os.path.join(artifacts, "data/question_ids/all", f"{name}_correct.csv"), ndmin=1, dtype=int)


def main():
    args = parse_args()
    groups = {}
    for k in args.feature_keys:
        groups[k] = Z.load_features(args.features_file, k)[0]
    if args.groups_json:
        for k, v in json.load(open(args.groups_json)).items():
            groups[k] = [int(x) for x in v]
    names = list(groups)
    out_dir = args.results_dir
    out_path = os.path.join(out_dir, f"gate_stats{'_' + args.tag if args.tag else ''}.npz")
    print(json.dumps({"groups": {k: len(v) for k, v in groups.items()}, "sae": args.sae_id, "out": out_path}, indent=1))
    if args.dry_run:
        print("DRY RUN OK (nothing loaded)")
        return

    import torch
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    import dsg_utils.dataset_utils as dataset_utils
    import evals.unlearning.utils.metrics as M
    from evals.unlearning.utils.var import GEMMA_INST_FORMAT

    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    artifacts = os.path.join(out_dir, "artifacts", Z.MODEL_NAME)
    model = HookedTransformer.from_pretrained_no_processing(Z.MODEL_NAME, device="cuda", dtype=torch.float32)
    sae = SAE.from_pretrained(release=Z.SAE_RELEASE, sae_id=args.sae_id, device="cuda")[0].to(device="cuda", dtype=torch.float32)
    hook_name = sae.cfg.hook_name
    layer = int(hook_name.split(".")[1])
    id_tensors = [torch.tensor(groups[n], device="cuda") for n in names]

    def rho_tokens(tokens):
        """tokens [b, T] (already with BOS at position 0, right-padded with pad id) -> rho [b, n_groups]"""
        _, cache = model.run_with_cache(tokens, names_filter=hook_name, stop_at_layer=layer + 1)   # blocks up to the SAE layer only
        acts = sae.encode(cache[hook_name])
        acts[:, 0, :] = 0.0
        valid = (tokens != model.tokenizer.pad_token_id)
        valid[:, 0] = True                       # BOS is part of the prompt (counted in T, never active)
        res = []
        for ids in id_tensors:
            pos = (acts[:, :, ids] > 0).any(dim=2) & valid
            res.append(pos.sum(1) / valid.sum(1))
        return torch.stack(res, dim=1).float().cpu().numpy()

    def rho_prompts(prompts, bs=8):
        out = []
        for i in range(0, len(prompts), bs):
            toks = model.to_tokens(prompts[i:i + bs], padding_side="right", prepend_bos=False).to("cuda")
            out.append(rho_tokens(toks))
        return np.concatenate(out) if out else np.zeros((0, len(names)))

    def rho_chunks(tokens, bs=16):
        return np.concatenate([rho_tokens(tokens[i:i + bs].to("cuda")) for i in range(0, len(tokens), bs)])

    def mcq_prompts(dataset_name, subset=None, variant="orig"):
        return PV.build(dataset_name, subset, variant)[0]

    sets, sizes = {}, {}
    want = lambda s: args.sets is None or s in args.sets or any(s.startswith(w) for w in args.sets)

    def record(name, rho):
        sets[name] = rho
        print(f"{name:42s} n={len(rho):5d}  mean rho per group: " + "  ".join(f"{n}={rho[:, j].mean():.3f}" for j, n in enumerate(names)), flush=True)

    t0 = time.time()
    # --- WikiText chunks
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split, tag in [("test", "test"), ("validation", "val")]:
        texts = [str(x["text"]) for x in raw[split] if len(x["text"]) > 50]
        for L in (1024, 128):
            if want(f"wiki_{tag}_{L}"):
                record(f"wiki_{tag}_{L}", rho_chunks(dataset_utils.tokenize_and_concat_dataset(model.tokenizer, texts, seq_len=L)))
    # --- MCQ prompts
    if want("mcq_cal"):
        record("mcq_cal", rho_prompts(sum([mcq_prompts(s) for s in CAL_SUBJECTS], [])))
    for s in EVAL_MMLU:
        if want(f"mcq_eval_{s}"):
            record(f"mcq_eval_{s}", rho_prompts(mcq_prompts(s, selected_ids(artifacts, s))))
    if want("mcq_bio_neighbours"):
        for s in BIO_NEIGHBOURS:
            record(f"mcq_bio_neighbours|{s}", rho_prompts(mcq_prompts(s)))
    if want("mcq_cyber_neighbours"):
        for s in CYBER_NEIGHBOURS:
            record(f"mcq_cyber_neighbours|{s}", rho_prompts(mcq_prompts(s)))
    ids = {d: selected_ids(artifacts, d) for d in ("wmdp-bio", "wmdp-cyber")}
    for d in ("wmdp-bio", "wmdp-cyber"):
        if want(d):
            record(d, rho_prompts(mcq_prompts(d, ids[d])))
        for v in VARIANTS:
            if want(f"{d}|{v}"):
                record(f"{d}|{v}", rho_prompts(mcq_prompts(d, ids[d], variant=v)))
    if want("mixed_bio_cyber"):
        pb, pc = mcq_prompts("wmdp-bio", ids["wmdp-bio"]), mcq_prompts("wmdp-cyber", ids["wmdp-cyber"])
        rng = random.Random(args.seed)
        mixed = []
        for i in range(len(pc)):
            b = pb[rng.randrange(len(pb))]
            # put the bio question and the cyber question into one user turn (chat template of the first prompt kept)
            body_b = b.split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
            body_c = pc[i].split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
            mixed.append(GEMMA_INST_FORMAT.format(prompt=body_b + "\n\n" + body_c) + "Answer: (")
        record("mixed_bio_cyber", rho_prompts(mixed))
    # --- forget-corpus text
    if any(want(s) for s in ("bio_forget_corpus", "cyber_forget_corpus")):
        corp = {}
        data_file = os.path.join(ROOT, "src/dynamic_sae_guardrails/evals/unlearning/data/bio-forget-corpus.jsonl")
        if os.path.exists(data_file):
            corp["bio_forget_corpus"] = [json.loads(l)["text"] for l in open(data_file) if len(json.loads(l)["text"]) > 50]
        else:
            print("bio-forget-corpus.jsonl not found, skipped")
        cy = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
        corp["cyber_forget_corpus"] = [t for t in cy["text"] if len(t) > 50]
        for name, texts in corp.items():
            random.Random(args.seed).shuffle(texts)
            for L in (1024, 128):
                if want(f"{name}_{L}") or want(name):
                    tok = dataset_utils.tokenize_and_concat_dataset(model.tokenizer, texts[:2000], seq_len=L)
                    sel = torch.randperm(len(tok), generator=torch.Generator().manual_seed(args.seed))[:args.n_corpus]
                    record(f"{name}_{L}", rho_chunks(tok[sel]))

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, groups=np.array(names), group_sizes=np.array([len(groups[n]) for n in names]),
                        **{"set|" + k: v for k, v in sets.items()})
    json.dump({"groups": groups, "sae": args.sae_id, "argv": sys.argv, "elapsed_s": time.time() - t0},
              open(out_path.replace(".npz", ".json"), "w"))
    print("written", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

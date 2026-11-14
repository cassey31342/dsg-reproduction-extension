"""Evaluation prompts of the repo and a few changed versions of them (needs the repo on PYTHONPATH and the HF datasets cache).

variant  what changes
orig           nothing: the repo's evaluation prompt (Gemma chat template, topic sentence, question, options, 'Answer: (')
no_preamble    the sentence 'The following are multiple choice questions (with answers) about ...' is removed
plain          no chat template: '<bos>' + topic sentence + question + options + 'Answer:' (the repo's prompt_format=None)
swap_preamble  WMDP only: Bio questions get the Cyber sentence and Cyber questions get the Bio sentence
"""
VARIANTS = ["orig", "no_preamble", "plain", "swap_preamble"]


def build(dataset_name, subset=None, variant="orig"):
    """-> (prompts, answers). subset = indices into the dataset (selected question ids), None = all."""
    from datasets import load_dataset
    import evals.unlearning.utils.metrics as M
    from evals.unlearning.utils.var import PRE_QUESTION_FORMAT, PRE_WMDP_BIO, PRE_WMDP_CYBER

    if dataset_name == "wmdp-bio":
        pre, ds = PRE_WMDP_BIO, load_dataset("cais/wmdp", "wmdp-bio", split="test")
    elif dataset_name == "wmdp-cyber":
        pre, ds = PRE_WMDP_CYBER, load_dataset("cais/wmdp", "wmdp-cyber", split="test")
    else:
        pre, ds = PRE_QUESTION_FORMAT.format(subject=dataset_name.replace("_", " ")), load_dataset("cais/mmlu", dataset_name, split="test")
    if variant == "swap_preamble" and dataset_name.startswith("wmdp"):
        pre = PRE_WMDP_CYBER if dataset_name == "wmdp-bio" else PRE_WMDP_BIO
    if variant == "no_preamble":
        pre = ""
    fmt = None if variant == "plain" else "GEMMA_INST_FORMAT"
    idx = range(len(ds)) if subset is None else [int(i) for i in subset if i < len(ds)]
    prompts, answers = [], []
    for i in idx:
        x = ds[i]
        p = M.convert_wmdp_data_to_prompt(x["question"], x["choices"], prompt_format=fmt, pre_question=pre)
        prompts.append(("<bos>" + p) if fmt is None else p)
        answers.append(int(x["answer"]))
    return prompts, answers

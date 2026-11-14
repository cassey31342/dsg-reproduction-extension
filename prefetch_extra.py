"""Download the extra MMLU subjects used by gate_stats.py into the local HF cache (run once on the login node, which has internet;
the compute jobs are offline).  python zero_shot/prefetch_extra.py
"""
import os
import sys

os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_stats import CAL_SUBJECTS, BIO_NEIGHBOURS, CYBER_NEIGHBOURS  # noqa: E402
from datasets import load_dataset  # noqa: E402

for s in CAL_SUBJECTS + BIO_NEIGHBOURS + CYBER_NEIGHBOURS:
    ds = load_dataset("cais/mmlu", s, split="test")
    print(s, len(ds), flush=True)
print("PREFETCH DONE")

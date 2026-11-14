"""Checks that the dual-gate hook is a faithful extension of the repo's clamp function. Runs on CPU, needs torch and the repo on the path.

    PYTHONPATH=<repo>/src/dynamic_sae_guardrails python zero_shot/test_dual_gate.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_gate import make_dual_gate_hook  # noqa: E402
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features as REPO  # noqa: E402

torch.manual_seed(0)
D, F, B, S, MULT = 16, 64, 6, 20, 500.0


class MockSAE:  # small ReLU SAE with the encode/decode interface used by the hook
    def __init__(self):
        self.W = torch.randn(D, F)
        self.b = torch.randn(F) * 0.3 - 4.0   # sparse SAE: features fire rarely, so prompts differ in how many tokens activate them
        self.Wd = torch.randn(F, D) * 0.2
        self.bd = torch.randn(D) * 0.1

    def encode(self, x):
        return torch.relu(x @ self.W + self.b)

    def decode(self, a):
        return a @ self.Wd + self.bd


sae, hook = MockSAE(), make_dual_gate_hook()
resid = torch.randn(B, S, D) * torch.linspace(0.3, 3.0, B).view(B, 1, 1)  # prompts with very different activity
ids1, ids2 = list(range(0, 8)), list(range(20, 30))
acts = sae.encode(resid).clone(); acts[:, 0, :] = 0
rate = lambda ids: ((acts[:, :, ids] > 0).sum(2) > 0).float().mean(1)
r1, r2 = rate(ids1), rate(ids2)
print("share of active tokens per prompt, group 1:", [round(x, 2) for x in r1.tolist()])
assert float(r1.max() - r1.min()) > 0.2 and float(r2.max() - r2.min()) > 0.2, "test data must make the gate fire for some prompts only"
tau1 = float(r1.median())
call = lambda groups, mode="per_domain": hook(resid.clone(), None, sae, [i for g, _ in groups for i in g], MULT, {"mode": mode, "groups": groups})
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


# 1. one group alone == the repo's function, for several thresholds (gate fires for some prompts and not for others)
for tau in (tau1, 0.0, 0.99):
    ref = REPO(resid.clone(), None, sae, ids1, MULT, tau)
    check(f"single group equals repo function (tau={tau:.2f})", torch.allclose(call([(ids1, tau)]), ref, atol=1e-5))
# 2. no gate fires -> residual unchanged
check("no gate fires: residual unchanged", torch.allclose(call([(ids1, 2.0), (ids2, 2.0)]), resid, atol=1e-5))
# 3. only one gate fires: equals that group alone
check("only group 1 fires = group 1 alone", torch.allclose(call([(ids1, tau1), (ids2, 2.0)]), call([(ids1, tau1)]), atol=1e-5))
check("only group 2 fires = group 2 alone", torch.allclose(call([(ids1, 2.0), (ids2, float(r2.median()))]), call([(ids2, float(r2.median()))]), atol=1e-5))
# 4. both fire (per_domain): changes at least as much as either alone
both = call([(ids1, 0.0), (ids2, 0.0)])
check("both fire: output differs from input", not torch.allclose(both, resid, atol=1e-5))
# 5. any_all: if neither fires, unchanged; if one fires, features of the other group are clamped too
check("any_all, neither fires: unchanged", torch.allclose(call([(ids1, 2.0), (ids2, 2.0)], "any_all"), resid, atol=1e-5))
a_any = call([(ids1, 0.0), (ids2, 2.0)], "any_all")
a_dom = call([(ids1, 0.0), (ids2, 2.0)], "per_domain")
check("any_all with one firing gate clamps more than per_domain", (a_any - resid).abs().sum() > (a_dom - resid).abs().sum())
# 6. overlapping feature lists: clamping is an assignment, so a shared feature is clamped once, at -MULT, if either gate fires there
ids_a, ids_b = list(range(0, 12)), list(range(8, 20))
ra, rb = rate(ids_a), rate(ids_b)
ta, tb = float(ra.median()), float(rb.median())


def reference(groups):
    a = sae.encode(resid.clone()); a[:, 0, :] = 0.0
    err = resid - sae.decode(a)
    new = a.clone()
    for b in range(B):
        for ids, tau in groups:
            active = [(a[b, t, ids] > 0).any().item() for t in range(S)]
            if torch.tensor(sum(active) / S, dtype=torch.float32) > torch.tensor(tau, dtype=torch.float32):   # float32 like the hook
                for t in range(S):
                    if active[t]:
                        new[b, t, ids] = -MULT
    return sae.decode(new) + err


check("overlapping lists, per domain: equals explicit loop", torch.allclose(call([(ids_a, ta), (ids_b, tb)]), reference([(ids_a, ta), (ids_b, tb)]), atol=1e-4))
check("overlapping lists, tau 0: equals explicit loop", torch.allclose(call([(ids_a, 0.0), (ids_b, 0.0)]), reference([(ids_a, 0.0), (ids_b, 0.0)]), atol=1e-4))
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)

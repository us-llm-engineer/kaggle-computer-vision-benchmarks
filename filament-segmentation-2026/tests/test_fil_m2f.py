"""Tests for the Mask2Former wrapper (needs transformers and network access to the public checkpoint). Run: python3 helpers/tests/test_fil_m2f.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fil_m2f as W  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print("ok  ", name)
        return fn
    return deco


torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = W.M2F().to(dev)
H = 128


def toy(B=2):
    x = torch.randn(B, 1, H, H, device=dev)
    lab = torch.zeros(B, H, H, dtype=torch.uint8, device=dev)
    lab[0, 20:40, 10:100] = 1; lab[0, 70:90, 30:60] = 2; lab[1, 50:70, 20:110] = 1
    n = torch.tensor([2, 1])
    return x, lab, n


@check("param groups: every parameter in exactly one of the four groups")
def _():
    g = W.param_groups_m2f(model)
    ids = [id(p) for v in g.values() for p in v]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())) and set(g) == {"encoder", "pixel_decoder", "transformer", "heads"}


@check("forward: logit map has the input size, the semantic probability is in [0,1]")
def _():
    x, _, _ = toy()
    model.eval()
    with torch.no_grad():
        lg = model(x)
    assert lg.shape == (2, 1, H, H) and torch.isfinite(lg).all()
    p = torch.sigmoid(lg)
    assert float(p.min()) >= 0 and float(p.max()) <= 1


@check("semantic_prob maths: certain filament query with an all-on mask gives ~1, 'no object' query gives ~0")
def _():
    cls = torch.tensor([[[10.0, -10.0]]])                 # one query, class 0 (filament) certain
    m = torch.full((1, 1, 8, 8), 10.0)
    assert float(W.M2F.semantic_prob(cls, m, (16, 16)).min()) > 0.99
    cls2 = torch.tensor([[[-10.0, 10.0]]])                # no-object certain
    assert float(W.M2F.semantic_prob(cls2, m, (16, 16)).max()) < 0.01


@check("training: loss finite, gradients reach the backbone, pixel decoder and transformer decoder, and overfitting one toy batch lowers the loss")
def _():
    model.train()
    x, lab, n = toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    first = last = None
    for it in range(10):
        loss, sem = model.train_step_outputs(x, lab, n)
        assert torch.isfinite(loss) and sem.shape == (2, 1, H, H)
        opt.zero_grad(); loss.backward()
        if it == 0:
            for k, ps in W.param_groups_m2f(model).items():
                assert sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None) > 0, k
        opt.step()
        first = float(loss) if first is None else first; last = float(loss)
    assert last < first, (first, last)


@check("empty target (an image crop without filaments) is accepted")
def _():
    model.train()
    x, lab, n = toy()
    lab[1] = 0; n = torch.tensor([2, 0])
    loss, _ = model.train_step_outputs(x, lab, n)
    assert torch.isfinite(loss)

print(f"{len(PASSED)} checks passed")

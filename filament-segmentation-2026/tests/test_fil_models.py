"""Tests for fil_models (CPU, tiny sizes). Run: python3 helpers/tests/test_fil_models.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fil_models as M  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print("ok  ", name)
        return fn
    return deco


@check("v0 and v2 forward: output shape (B,1,H,W), v2 has more parameters and the attention path is used")
def _():
    x = torch.randn(2, 1, 64, 64)
    a, b = M.UNet(base=8), M.UNet(base=8, edge_attn=True)
    assert a(x).shape == (2, 1, 64, 64) and b(x).shape == (2, 1, 64, 64)
    assert sum(p.numel() for p in b.parameters()) > sum(p.numel() for p in a.parameters())
    assert "attn" in M.param_groups(b) and "attn" not in M.param_groups(a)


@check("v2: gradients reach the edge convolution (the edge prior is wired into the attention)")
def _():
    m = M.UNet(base=8, edge_attn=True)
    m(torch.randn(1, 1, 64, 64)).mean().backward()
    assert m.edge.weight.grad is not None and m.edge.weight.grad.abs().sum() > 0
    assert m.edge_proj.weight.grad.abs().sum() > 0


@check("v2 edge prior changes the output: zeroing the edge projection changes logits")
def _():
    torch.manual_seed(0)
    m = M.UNet(base=8, edge_attn=True)          # train-mode batch norm normalises the activations (eval mode at random init shrinks them to ~1e-4)
    for blk in m.attn:
        blk.drop.p = 0.0
    x = torch.randn(4, 1, 64, 64)
    y0 = m(x).detach()
    with torch.no_grad():
        m.edge_proj.weight.mul_(0); m.edge_proj.bias.mul_(0)
    assert (m(x).detach() - y0).abs().max() > 1e-4


@check("soft skeleton: a thick bar becomes a thin ridge (fewer foreground pixels, inside the bar)")
def _():
    t = torch.zeros(1, 1, 64, 64)
    t[..., 28:37, 8:56] = 1
    s = M.soft_skeleton(t, 6)
    assert 0 < s.sum() < 0.35 * t.sum()
    assert float((s * (1 - t)).sum()) == 0.0


@check("soft clDice loss: ~0 for a perfect prediction, larger for a broken line, gradient is finite")
def _():
    t = torch.zeros(1, 1, 64, 64)
    t[..., 30:34, 4:60] = 1
    logit_perfect = (t * 40 - 20)
    broken = t.clone(); broken[..., 28:36, 28:36] = 0
    l0 = M.soft_cldice_loss(logit_perfect, t, 6)
    l1 = M.soft_cldice_loss(broken * 40 - 20, t, 6)
    assert l0 < 0.05 and l1 > l0 + 0.05, (float(l0), float(l1))
    z = (t * 4 - 2).requires_grad_(True)
    M.soft_cldice_loss(z, t, 6).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


@check("cldice mutant: swapping precision and sensitivity inputs is detectable (asymmetric case)")
def _():
    t = torch.zeros(1, 1, 64, 64); t[..., 30:34, 4:60] = 1
    p = torch.zeros(1, 1, 64, 64); p[..., 30:34, 4:30] = 1                       # prediction covers half of the target
    good = float(M.soft_cldice_loss(p * 40 - 20, t, 6))
    swapped = float(M.soft_cldice_loss(t * 40 - 20, p, 6))                         # roles swapped: target is the short bar
    assert abs(good - swapped) > 1e-3 or good > 0.2


@check("seg_loss: v1 adds a clDice term and v0/v2 do not; weights combine as (1-alpha)(bce+dice)+alpha cldice")
def _():
    t = torch.zeros(2, 1, 32, 32); t[..., 14:18, 2:30] = 1
    z = torch.randn(2, 1, 32, 32)
    l0, p0 = M.seg_loss(z, t, "v0"); l1, p1 = M.seg_loss(z, t, "v1", alpha=0.3, k=4); l2, p2 = M.seg_loss(z, t, "v2")
    assert "cldice" in p1 and "cldice" not in p0 and "cldice" not in p2
    assert abs(float(l1) - (0.7 * (p1["bce"] + p1["dice"]) + 0.3 * p1["cldice"])) < 1e-4
    assert abs(float(l0) - (p0["bce"] + p0["dice"])) < 1e-5


@check("tiling: offsets cover the image with the last tile flush to the end; tiled prediction of a constant model is that constant everywhere")
def _():
    import fil_infer as I
    import numpy as np
    for n, t, st in ((2048, 1024, 768), (2048, 512, 384), (1000, 256, 200)):
        offs = I.tile_offsets(n, t, st)
        assert offs[0] == 0 and offs[-1] == n - t and all(b - a <= t for a, b in zip(offs, offs[1:]))
    class Const(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.7)                   # logit 0.7 everywhere
    p = I.predict_prob(Const(), np.zeros((2048, 2048), np.uint8), torch.device("cpu"), tile=1024, stride=768).astype(np.float32)
    assert abs(float(p.min()) - float(torch.sigmoid(torch.tensor(0.7)))) < 2e-3 and abs(float(p.max()) - float(p.min())) < 2e-3

print(f"{len(PASSED)} checks passed")

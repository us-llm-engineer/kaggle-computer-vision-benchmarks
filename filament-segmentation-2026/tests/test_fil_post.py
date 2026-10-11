"""Tests for the loop-2 post-processing and models. Run: python3 tests/test_fil_post.py   (CPU, well under 90 s; pre-trained weights are not downloaded)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fil_audit as A  # noqa: E402
import fil_common as F  # noqa: E402
import fil_post as Q  # noqa: E402
import fil_pre as P  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print("ok  ", name)
        return fn
    return deco


rng = np.random.default_rng(0)
S = 128
base = rng.random((S, S)).astype(np.float32) * 0.2
prob = base.copy()
prob[10:20, 10:60] = 0.9                       # a confident bar
prob[50:54, 80:120] = 0.55                     # a weak bar (mean 0.55)
prob[100:103, 20:23] = 0.9                     # a tiny confident blob (9 px)
prob[70:80, 10:12] = 0.45                      # sub-threshold ridge
prob[70:72, 10:12] = 0.95                      # ... containing a seed


@check("with lo=None, min_conf=0 the new recovery equals the round-1 reference")
def _():
    for thr, ma, cr in ((0.5, 30, 0), (0.4, 10, 2), (0.6, 50, 0)):
        a = Q.instances_post(prob, thr, None, ma, 0.0, cr)
        b = F.instances_from_prob(prob, thr, ma, cr)
        assert np.array_equal(a > 0, b > 0) and a.max() == b.max()


@check("confidence filter removes a planted weak object and keeps the confident one")
def _():
    a = Q.instances_post(prob, 0.5, None, 30, 0.0, 0)
    b = Q.instances_post(prob, 0.5, None, 30, 0.7, 0)
    assert a[52, 100] > 0 and b[52, 100] == 0 and b[15, 30] > 0


@check("hysteresis grows a seeded component into the low-threshold ridge and drops unseeded low regions")
def _():
    one = Q.instances_post(prob, 0.9, None, 5, 0.0, 0)
    hys = Q.instances_post(prob, 0.9, 0.4, 5, 0.0, 0)
    assert one[75, 10] == 0 and hys[75, 10] > 0               # ridge pixel only reached through the seed
    assert hys[52, 100] == 0                                  # weak bar (0.55) has no seed above 0.9: dropped


@check("area filter removes the 9-pixel blob")
def _():
    assert Q.instances_post(prob, 0.5, None, 30)[101, 21] == 0 and Q.instances_post(prob, 0.5, None, 5)[101, 21] > 0


@check("ensemble of identical maps is the map")
def _():
    m = prob.astype(np.float16)
    assert np.allclose(Q.ensemble([m, m, m]).astype(np.float32), m.astype(np.float32), atol=1e-3)


@check("stitching merges one object seen from two tiles and keeps two separate objects")
def _():
    a = np.zeros((40, 40), bool); a[5:15, 5:35] = True
    t1 = (0.9, 0, 0, a)
    b = np.zeros((40, 40), bool); b[5:15, 0:20] = True          # same object in a tile shifted by 20 px: global columns 20..39 (overlaps a on 20..34)
    t2 = (0.8, 0, 20, b)
    c = np.zeros((40, 40), bool); c[30:36, 5:35] = True
    t3 = (0.7, 0, 0, c)
    lab = Q.stitch_instances([t1, t2, t3], 64, min_area=5)
    assert len(np.unique(lab[lab > 0])) == 2
    assert lab[10, 38] == lab[10, 10] > 0                  # columns 35..39 exist only in the second tile and joined the first label


@check("ResNet34 U-Net: shapes, 1-channel input, edge variant changes the output")
def _():
    torch.manual_seed(0)
    m = P.ResNet34UNet(pretrained=False).eval()
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    e = P.ResNet34UNet(pretrained=False, edge_attn=True).train()
    e.edge_proj.weight.data.normal_()
    y1 = e(x)
    e.edge_proj.weight.data.zero_()
    assert not torch.allclose(y1, e(x))
    g = P.param_groups_r34(m)
    assert sum(len(v) for v in g.values()) == len(list(m.parameters()))


@check("EMA: update matches a hand computation and never aliases the model")
def _():
    lin = torch.nn.Linear(3, 1)
    ema = P.EMA(lin, 0.9)
    w0 = ema.model.weight.detach().clone()
    lin.weight.data.add_(1.0)
    ema.update(lin)
    assert torch.allclose(ema.model.weight, 0.9 * w0 + 0.1 * lin.weight.detach(), atol=1e-6)
    lin.weight.data.add_(5.0)
    assert not torch.allclose(ema.model.weight, lin.weight)


@check("FP audit: an unmatched prediction that another copy drew is counted as a sibling match")
def _():
    own = np.zeros((40, 40), np.int32); own[2:8, 2:30] = 1
    sib = np.zeros((40, 40), np.int32); sib[2:8, 2:30] = 1; sib[20:26, 2:30] = 2
    pred = np.zeros((40, 40), np.int32); pred[2:8, 2:30] = 1; pred[20:26, 2:30] = 2; pred[34:38, 2:10] = 3
    r = A.fp_audit(pred, own, [sib])
    assert (r["pred"], r["tp"], r["fp"], r["fp_in_sibling"]) == (3, 1, 2, 1), r


print(f"{len(PASSED)} checks passed")

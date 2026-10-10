"""Network definitions shared by training (train_cnn.py) and the figure script (make_figures.py)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fn_common as C


class Block(nn.Module):
    def __init__(s, i, o, stride):
        super().__init__(); s.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(o); s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.skip = None if (i == o and stride == 1) else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        y = s.b2(s.c2(F.relu(s.b1(s.c1(x))))); return F.relu(y + (x if s.skip is None else s.skip(x)))


def trunk(widths):
    layers = [nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)]; prev = 32
    for w in widths: layers += [Block(prev, w, 2), Block(w, w, 1)]; prev = w
    return nn.Sequential(*layers), prev


class Pooled(nn.Module):
    """Trunk + (average, max) pooling + dropout + linear head over the 290 categories."""

    def __init__(s, widths):
        super().__init__(); s.trunk, ch = trunk(widths); s.drop = nn.Dropout(0.3); s.fc = nn.Linear(2 * ch, C.NCLS); s.dim = 2 * ch

    def features(s, x):
        f = s.trunk(x); return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)

    def forward(s, x):
        f = s.features(x); return s.fc(s.drop(f)), f


class TileMIL(nn.Module):
    """Shared tile encoder over a 2x4 grid of 128x112 tiles; image logit = (1/r) log mean exp(r * tile logit)."""

    def __init__(s, widths, r=3.0):
        super().__init__(); s.enc = Pooled(widths); s.r = r; s.dim = s.enc.dim

    def tiles(s, x):
        B = x.shape[0]; return x.unfold(2, 128, 128).unfold(3, 112, 112).permute(0, 2, 3, 1, 4, 5).reshape(B * 8, 3, 128, 112)

    def forward(s, x):
        B = x.shape[0]; logit, feat = s.enc(s.tiles(x)); logit = logit.view(B, 8, -1)
        return (torch.logsumexp(s.r * logit, 1) - math.log(8)) / s.r, feat.view(B, 8, -1).mean(1)


def build(kind):
    return Pooled([64, 128, 256]) if kind == "cnn" else TileMIL([48, 96, 160])

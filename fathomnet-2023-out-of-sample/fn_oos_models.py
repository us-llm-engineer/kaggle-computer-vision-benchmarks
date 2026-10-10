"""Networks of the out-of-sample round: the residual trunk of the first attempt with a variant head, plus a 20-way supercategory head used by the
ranking tail. kinds: 'bce' (V0 control and V1/V2 trunk), 'evid' (V3, Beta evidential two-branch head)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import fn_common as C
import fn_models as M


class OOSNet(nn.Module):
    def __init__(s, kind="bce", widths=(64, 128, 256), hidden=256, n_super=20, n_shells=4):
        super().__init__()
        s.kind = kind; s.trunk, ch = M.trunk(list(widths)); s.dim = 2 * ch; s.drop = nn.Dropout(0.3)
        s.fc = nn.Linear(s.dim, C.NCLS)                                              # category logits (bce kinds)
        s.sc = nn.Linear(s.dim, n_super)                                              # supercategory head (shared tail module)
        s.radius = nn.Linear(s.dim, n_shells)                                         # BootOOD radius head (V2 only, discarded at test time)
        if kind == "evid":
            mk = lambda: nn.Sequential(nn.Linear(s.dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, C.NCLS))
            s.pos, s.neg = mk(), mk()

    def features(s, x):
        f = s.trunk(x); return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)

    def heads(s, f):
        """Everything the training and evaluation code needs from a feature vector f."""
        out = {"sc": s.sc(s.drop(f))}
        if s.kind == "evid":
            fp, fn = s.pos(s.drop(f)), s.neg(s.drop(f))
            alpha, beta = F.elu(fp.float()) + 2.0, F.elu(fn.float()) + 2.0
            p = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)
            out.update(alpha=alpha, beta=beta, logit=torch.log(p / (1 - p)))
        else:
            out["logit"] = s.fc(s.drop(f))
        return out

    def forward(s, x):
        f = s.features(x); out = s.heads(f); out["feat"] = f; return out


def beta_risk_loss(alpha, beta, y, pos_weight=None):
    """Bayes risk of BCE under Beta(alpha, beta); the positive term can carry the same clipped weights as the BCE control."""
    pos = y * (torch.digamma(alpha + beta) - torch.digamma(alpha))
    if pos_weight is not None: pos = pos * pos_weight
    neg = (1 - y) * (torch.digamma(alpha + beta) - torch.digamma(beta))
    return (pos + neg).mean()

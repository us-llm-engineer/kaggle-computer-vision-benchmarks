"""Models and losses for the filament task: a U-Net (V0 control), the same U-Net with edge-guided bottleneck attention (V2), and the loss terms
(BCE + soft Dice, optional soft-clDice for V1). Written for torch >= 2.1; all heavy ops run on the device of the input.

Sources: U-Net layout and edge-guided attention after the barb-aware filament segmentation paper (edge map = sigmoid(conv3x3(input)), resized to the
bottleneck, projected by a 1x1 convolution and ADDED to the attention queries and keys, values untouched, no positional encoding, 2 blocks, 4 heads);
soft skeleton and soft clDice after the topology-preserving loss paper (iterated min/max pooling, k >= maximum structure radius, loss
(1-alpha)(1-softDice) + alpha(1-softclDice) in the paper; here the base term is BCE + soft Dice, a stated deviation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dconv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EGMHSA(nn.Module):
    """edge-guided multi-head self-attention on the bottleneck: Q = Wq(x + E'), K = Wk(x + E'), V = Wv(x); residual + LayerNorm + dropout."""

    def __init__(self, c, heads=4, drop=0.1):
        super().__init__()
        assert c % heads == 0
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(c, c) for _ in range(4))
        self.norm = nn.LayerNorm(c)
        self.drop = nn.Dropout(drop)

    def forward(self, x, e):                        # x, e: (B, N, C)
        B, N, C = x.shape
        sp = lambda t: t.view(B, N, self.h, C // self.h).transpose(1, 2)
        q, k, v = sp(self.q(x + e)), sp(self.k(x + e)), sp(self.v(x))
        a = F.scaled_dot_product_attention(q, k, v)      # softmax(QK^T / sqrt(d)) V, fused kernel
        a = self.o(a.transpose(1, 2).reshape(B, N, C))
        return self.norm(x + self.drop(a))


class UNet(nn.Module):
    """4-stage U-Net (/16 at the bottleneck). edge_attn=True inserts two EG-MHSA blocks after the bottleneck convolution (variant V2)."""

    def __init__(self, base=32, edge_attn=False, heads=4):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        cin = 1
        for c in ch:
            self.enc.append(dconv(cin, c)); cin = c
        self.bott = dconv(ch[-1], ch[-1] * 2)
        cb = ch[-1] * 2
        self.edge_attn = edge_attn
        if edge_attn:
            self.edge = nn.Conv2d(1, 1, 3, padding=1)
            self.edge_proj = nn.Conv2d(1, cb, 1)
            self.attn = nn.ModuleList([EGMHSA(cb, heads) for _ in range(2)])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cur = cb
        for c in reversed(ch):
            self.up.append(nn.ConvTranspose2d(cur, c, 2, stride=2))
            self.dec.append(dconv(c * 2, c)); cur = c
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):                           # x: (B,1,H,W), H and W multiples of 16
        skips = []
        h = x
        for blk in self.enc:
            h = blk(h); skips.append(h); h = F.max_pool2d(h, 2)
        h = self.bott(h)
        if self.edge_attn:
            e = torch.sigmoid(self.edge(x))                                         # (B,1,H,W) learned edge map from the input
            e = self.edge_proj(F.interpolate(e, size=h.shape[-2:], mode="bilinear", align_corners=False))
            B, C, H, W = h.shape
            t, et = h.flatten(2).transpose(1, 2), e.flatten(2).transpose(1, 2)
            for blk in self.attn:
                t = blk(t, et)
            h = t.transpose(1, 2).reshape(B, C, H, W)
        for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), sk], 1))
        return self.head(h)                         # logits (B,1,H,W)


def param_groups(model):
    """name -> parameters, for per-group gradient / update statistics."""
    g = {"enc": [], "bott": [], "attn": [], "dec": [], "head": []}
    for n, p in model.named_parameters():
        key = "attn" if n.startswith(("attn", "edge")) else n.split(".")[0]
        key = {"up": "dec"}.get(key, key)
        g.setdefault(key, []).append(p)
    return {k: v for k, v in g.items() if v}


# ---------------------------------------------------------------------------------------------------------------- losses
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


def _erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, k):
    """differentiable skeleton by iterated soft opening, k iterations (k >= the largest structure radius in pixels)."""
    s = F.relu(x - _dilate(_erode(x)))
    for _ in range(k):
        x = _erode(x)
        delta = F.relu(x - _dilate(_erode(x)))
        s = s + F.relu(delta - s * delta)
    return s


def soft_cldice_loss(logits, target, k=15, eps=1e-6):
    """1 - soft-clDice. Tprec uses the predicted skeleton against the target mask, Tsens the target skeleton against the prediction
    (definitions as read from the paper: Tprec = |S_P . V_L| / |S_P|, Tsens = |S_L . V_P| / |S_L|); the target skeleton needs no gradient."""
    p = torch.sigmoid(logits.float())
    sp = soft_skeleton(p, k)
    with torch.no_grad():
        sl = soft_skeleton(target, k)
    tprec = ((sp * target).sum() + eps) / (sp.sum() + eps)
    tsens = ((sl * p).sum() + eps) / (sl.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def seg_loss(logits, target, variant, alpha=0.3, k=15, pos_weight=None):
    """V0 / V2: BCE + soft Dice.  V1: (1 - alpha) (BCE + soft Dice) + alpha (1 - soft-clDice).
    returns (total, parts dict of DETACHED DEVICE TENSORS): no host synchronisation inside the training step (read the parts once per epoch)."""
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pw)
    dsc = soft_dice_loss(logits, target)
    parts = {"bce": bce.detach(), "dice": dsc.detach()}
    if variant == "v1":
        cl = soft_cldice_loss(logits, target, k)
        parts["cldice"] = cl.detach()
        return (1 - alpha) * (bce + dsc) + alpha * cl, parts
    return bce + dsc, parts

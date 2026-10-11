"""Loop-2 models: a U-Net with an ImageNet pre-trained ResNet34 encoder (TernausNet-style transfer, 1-channel input made by summing the first-layer filters),
optionally with the edge-guided bottleneck attention of the round-1 V2 model, plus the weight-EMA helper and the model factory for the new variants.

Variants (names used by train_fil2.py):
  m2fx  Mask2Former swin-tiny (COCO instance) continued from the round-1 checkpoint (warm restart)
  r34   ResNet34 U-Net, BCE + soft Dice, EMA of weights
  r34e  r34 + two edge-guided attention blocks at the 1/32 bottleneck
  r34c  r34 trained on the soft consensus of the annotator copies of one image (annotator-aware target)
Source for the transfer recipe: TernausNet (pre-trained encoder, loss BCE and Jaccard-type term; pre-trained models converge faster and settle higher);
encoder learning rate lower than the decoder's, as the fine-tuning guidance of the pre-training study."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as TF

import fil_models as M


def _gn_groups(c):
    for g in (16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def dec_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(cout), cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(cout), cout), nn.ReLU(inplace=True))


class ResNet34UNet(nn.Module):
    """encoder: torchvision ResNet34 (strides 2, 4, 8, 16, 32); decoder: transposed-conv up-sampling with skip concatenation and GroupNorm
    (small batches at 512 px), a last block at full resolution that also sees the raw input so 1-pixel detail is not lost at the 1/2 stem."""

    def __init__(self, pretrained=True, edge_attn=False, heads=4):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        w = r.conv1.weight.detach().sum(1, keepdim=True)                       # (64,1,7,7): summed RGB filters = response to a grey image replicated 3x
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.conv1.weight.data.copy_(w)
        self.bn1, self.relu, self.pool = r.bn1, r.relu, r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.edge_attn = edge_attn
        if edge_attn:
            self.edge = nn.Conv2d(1, 1, 3, padding=1)
            self.edge_proj = nn.Conv2d(1, 512, 1)
            self.attn = nn.ModuleList([M.EGMHSA(512, heads) for _ in range(2)])
        self.up4, self.d4 = nn.ConvTranspose2d(512, 256, 2, 2), dec_block(512, 256)      # /32 -> /16, skip l3
        self.up3, self.d3 = nn.ConvTranspose2d(256, 128, 2, 2), dec_block(256, 128)      # /16 -> /8,  skip l2
        self.up2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), dec_block(128, 64)        # /8  -> /4,  skip l1
        self.up1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), dec_block(32 + 64, 32)     # /4  -> /2,  skip stem
        self.up0, self.d0 = nn.ConvTranspose2d(32, 32, 2, 2), dec_block(32 + 1, 32)      # /2  -> /1,  skip raw input
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):                                                     # x (B,1,H,W), H and W multiples of 32
        s0 = self.relu(self.bn1(self.conv1(x)))                               # /2, 64
        s1 = self.l1(self.pool(s0))                                           # /4, 64
        s2 = self.l2(s1)                                                      # /8, 128
        s3 = self.l3(s2)                                                      # /16, 256
        h = self.l4(s3)                                                       # /32, 512
        if self.edge_attn:
            e = torch.sigmoid(self.edge(x))
            e = self.edge_proj(TF.interpolate(e, size=h.shape[-2:], mode="bilinear", align_corners=False))
            B, C, H, W = h.shape
            t, et = h.flatten(2).transpose(1, 2), e.flatten(2).transpose(1, 2)
            for blk in self.attn:
                t = blk(t, et)
            h = t.transpose(1, 2).reshape(B, C, H, W)
        h = self.d4(torch.cat([self.up4(h), s3], 1))
        h = self.d3(torch.cat([self.up3(h), s2], 1))
        h = self.d2(torch.cat([self.up2(h), s1], 1))
        h = self.d1(torch.cat([self.up1(h), s0], 1))
        h = self.d0(torch.cat([self.up0(h), x], 1))
        return self.head(h)


def param_groups_r34(model):
    g = {"enc": [], "attn": [], "dec": [], "head": []}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        if top in ("conv1", "bn1", "l1", "l2", "l3", "l4"):
            g["enc"].append(p)
        elif top in ("attn", "edge", "edge_proj"):
            g["attn"].append(p)
        elif top == "head":
            g["head"].append(p)
        else:
            g["dec"].append(p)
    return {k: v for k, v in g.items() if v}


def build_model2(cfg, pretrained=True):
    """variant -> model (new variants only; round-1 variants stay in fil_infer.build_model). pretrained=False builds the architecture for loading our own weights."""
    v = cfg["variant"]
    if v == "m2fx":
        import fil_m2f as W
        return W.M2F(pretrained=pretrained)
    if v in ("r34", "r34c"):
        return ResNet34UNet(pretrained=pretrained)
    if v == "r34e":
        return ResNet34UNet(pretrained=pretrained, edge_attn=True)
    raise ValueError(v)


class EMA:
    """exponential moving average of the weights (and a copy of the buffers). update() uses foreach kernels on the device and never reads a value back."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step=None):
        d = self.decay if step is None else min(self.decay, (1 + step) / (10 + step))       # warm-up so the first steps are not dominated by the initial weights
        ep, mp_ = [p.detach() for p in self.model.parameters()], [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ep, mp_, 1 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)


def load_run2(run, dev, which="best.pt"):
    """loop-2 run directory -> (eval-mode model with the EMA weights, config)."""
    import json
    import os
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = build_model2(cfg, pretrained=False).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, which), map_location=dev, weights_only=False)["model"])
    return model.eval(), cfg

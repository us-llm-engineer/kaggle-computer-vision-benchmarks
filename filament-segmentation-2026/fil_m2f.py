"""Mask2Former wrapper (variant V3): the smallest pre-trained Mask2Former (Swin-tiny, COCO instance) fine-tuned on one class, 'filament'.
Training uses the model's own mask-classification loss (Hungarian matching of queries to instances, masked attention, point-sampled mask losses).
Inference combines the queries into ONE semantic probability map, p(x) = sum_q P(class=filament | q) * sigmoid(mask_q(x)) (the standard MaskFormer
semantic combination), returned as a logit so that the same tiled prediction, instance recovery and PQ scoring as for the U-Net variants apply.
Source for the architecture and the loss: the universal-segmentation paper (masked attention, 100 queries, mask loss weights 5/5, class weight 2). The paper never
trains from scratch and does not evaluate small datasets or thin structures, so the choice of this checkpoint is ours (the rules allow external pre-trained models)."""
import torch
import torch.nn as nn

NAME = "facebook/mask2former-swin-tiny-coco-instance"


class M2F(nn.Module):
    def __init__(self, name=NAME, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, Mask2FormerForUniversalSegmentation
        kw = {"id2label": {0: "filament"}, "label2id": {"filament": 0}}
        if pretrained:
            self.hf = Mask2FormerForUniversalSegmentation.from_pretrained(name, ignore_mismatched_sizes=True, **kw)
        else:                                                    # architecture only (weights come from our own checkpoint)
            cfg = AutoConfig.from_pretrained(name, **kw)
            self.hf = Mask2FormerForUniversalSegmentation(cfg)

    @staticmethod
    def prep(x):
        """(B,1,H,W) preprocessed float image (roughly zero-mean, unit-scale) -> (B,3,H,W) as the pre-trained backbone expects."""
        return x.expand(-1, 3, -1, -1).contiguous()

    @staticmethod
    def semantic_prob(cls_logits, mask_logits, size):
        """(B,Q,2), (B,Q,h,w) -> (B,1,H,W) probability that a pixel belongs to a filament."""
        p = torch.softmax(cls_logits.float(), -1)[..., 0]                         # (B,Q): class 'filament' (index 0); index 1 is 'no object'
        m = torch.sigmoid(mask_logits.float())
        sem = torch.einsum("bq,bqhw->bhw", p, m).clamp(0, 1)[:, None]
        return torch.nn.functional.interpolate(sem, size=size, mode="bilinear", align_corners=False).clamp(0, 1)

    def forward(self, x):                                                          # inference: logit of the semantic probability
        o = self.hf(pixel_values=self.prep(x))
        p = self.semantic_prob(o.class_queries_logits, o.masks_queries_logits, x.shape[-2:])
        return torch.log(p + 1e-4) - torch.log(1 - p + 1e-4)

    def train_step_outputs(self, x, lab, n):
        """x (B,1,H,W); lab (B,H,W) uint8 with DENSE instance ids 1..n[b] (0 = background); n (B,) instance counts.
        returns (loss, semantic probability (B,1,H,W)). No host synchronisation here beyond what the model's own matching needs."""
        B = x.shape[0]
        masks, classes = [], []
        for b in range(B):
            nb = int(n[b])
            ids = torch.arange(1, nb + 1, device=x.device)
            masks.append((lab[b][None] == ids[:, None, None]).float() if nb else torch.zeros(0, *lab.shape[-2:], device=x.device))
            classes.append(torch.zeros(nb, dtype=torch.long, device=x.device))
        o = self.hf(pixel_values=self.prep(x), mask_labels=masks, class_labels=classes)
        return o.loss, self.semantic_prob(o.class_queries_logits.detach(), o.masks_queries_logits.detach(), x.shape[-2:])


def param_groups_m2f(model):
    """name -> parameters: backbone (Swin encoder), pixel decoder, transformer decoder, heads (class and mask embedding)."""
    g = {"encoder": [], "pixel_decoder": [], "transformer": [], "heads": []}
    for n, p in model.named_parameters():
        if "pixel_level_module.encoder" in n:
            g["encoder"].append(p)
        elif "pixel_level_module.decoder" in n:
            g["pixel_decoder"].append(p)
        elif "transformer_module" in n:
            g["transformer"].append(p)
        else:
            g["heads"].append(p)
    return {k: v for k, v in g.items() if v}

"""Loop-2 instance recovery and ensembling (new; fil_common.instances_from_prob stays the round-1 reference).
  instances_post   hysteresis + closing + area and mean-confidence filters -> instance label image
  ensemble         mean of probability maps (optionally with flipped-input maps computed elsewhere)
  native_from_m2f  Mask2Former query masks as instances (score = p_class x mean mask probability), stitched across overlapping tiles
Source of the ideas: removal of small spurious instances after hole filling (Deep Watershed Transform), removal of low-confidence matches and
confidence-ranked suppression (PanSR); both are applied here to our own probability maps, and every parameter is chosen on CHOOSE only."""
import os
import sys

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402


def instances_post(prob, hi=0.5, lo=None, min_area=30, min_conf=0.0, close_radius=0):
    """probability -> instance labels. lo < hi: hysteresis (a component of prob > lo survives only if it contains a pixel with prob > hi).
    Components are then dropped when their area is below `min_area` or their mean probability is below `min_conf`. lo = None means lo = hi
    and min_conf = 0 gives exactly fil_common.instances_from_prob."""
    prob = np.asarray(prob, dtype=np.float32)
    lo = hi if lo is None else min(lo, hi)
    m = prob > lo
    if lo < hi:
        lab0, n0 = ndi.label(m, structure=np.ones((3, 3)))
        if n0 == 0:
            return lab0.astype(np.int32)
        seeds = np.zeros(n0 + 1, bool)
        seeds[np.unique(lab0[prob > hi])] = True
        seeds[0] = False
        m = seeds[lab0]
    if close_radius > 0:
        k = int(close_radius)
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        m = ndi.binary_closing(m, structure=(yy ** 2 + xx ** 2) <= k * k)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return lab.astype(np.int32)
    areas = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    keep = areas >= min_area
    if min_conf > 0:
        mean_p = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n + 1) / np.maximum(areas, 1)
        keep &= mean_p >= min_conf
    keep[0] = False
    return (np.cumsum(keep) * keep)[lab].astype(np.int32)


def ensemble(maps):
    """mean of float16/32 probability maps (equal weights)."""
    acc = np.zeros(maps[0].shape, np.float32)
    for m in maps:
        acc += m.astype(np.float32)
    return (acc / len(maps)).astype(np.float16)


def stitch_instances(items, size, merge_frac=0.5, min_area=30):
    """items: list of (score, y0, x0, bool mask) from overlapping tiles. Greedy by descending score into one label image: a mask whose pixels already
    labelled cover >= merge_frac of its area is the same object seen from another tile (its free pixels are added to that label), otherwise it becomes
    a new instance on the free pixels. Returns int32 labels (0 = background)."""
    out = np.zeros((size, size), np.int32)
    nid = 0
    for score, y0, x0, m in sorted(items, key=lambda t: -t[0]):
        h, w = m.shape
        win = out[y0:y0 + h, x0:x0 + w]
        area = int(m.sum())
        if area < min_area:
            continue
        under = win[m]
        occ = under[under > 0]
        if occ.size >= merge_frac * area:
            lab = int(np.bincount(occ).argmax())
            win[m & (win == 0)] = lab
        else:
            nid += 1
            win[m & (win == 0)] = nid
    return out


def native_from_m2f(model, xq, dev, tile=512, stride=384, batch=4, mask_thr=0.5, score_thr=0.3, min_area=30):
    """full-image pass of the Mask2Former wrapper: returns (fused probability map float16, native instance labels int32 after stitching).
    Both come from the same forward passes, so they can be compared on identical inputs."""
    import torch
    import fil_infer as I
    x = torch.from_numpy(F.to_float(np.asarray(xq))).to(dev)
    S = x.shape[-1]
    offs = I.tile_offsets(S, tile, stride)
    coords = [(y, xx) for y in offs for xx in offs]
    acc = torch.zeros(S, S, device=dev); wsum = torch.zeros(S, S, device=dev)
    win = I._window(tile, dev)
    items = []
    ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            cs = coords[i:i + batch]
            t = torch.stack([x[y:y + tile, xx:xx + tile] for y, xx in cs])[:, None]
            with ac:
                o = model.hf(pixel_values=model.prep(t))
            cls = torch.softmax(o.class_queries_logits.float(), -1)[..., 0]                    # (B,Q)
            mq = torch.sigmoid(o.masks_queries_logits.float())                                  # (B,Q,h,w)
            sem = (cls[..., None, None] * mq).sum(1, keepdim=True).clamp(0, 1)
            sem = torch.nn.functional.interpolate(sem, size=(tile, tile), mode="bilinear", align_corners=False).clamp(0, 1)[:, 0]
            up = torch.nn.functional.interpolate(mq, size=(tile, tile), mode="bilinear", align_corners=False)
            binm = up > mask_thr
            msc = (up * binm).sum((-1, -2)) / binm.sum((-1, -2)).clamp_min(1)
            score = cls * msc
            for b, (y, xx) in enumerate(cs):
                acc[y:y + tile, xx:xx + tile] += sem[b] * win; wsum[y:y + tile, xx:xx + tile] += win
                for q in torch.nonzero(score[b] > score_thr).flatten().tolist():
                    items.append((float(score[b, q]), y, xx, binm[b, q].cpu().numpy()))
    return (acc / wsum).half().cpu().numpy(), stitch_instances(items, S, min_area=min_area)

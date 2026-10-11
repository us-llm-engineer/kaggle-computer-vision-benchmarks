"""False-positive audit: of the predicted instances that match nothing in the scored annotator copy (IoU > 0.5), how many match an instance of another
annotator copy of the same image? Those are real filaments the scored annotator did not draw (label disagreement), not model hallucinations."""
import numpy as np


def iou_pairs(gt, pred):
    """sparse IoU between every gt instance and every predicted instance that overlap: dict {(g, p): iou}; ids start at 1."""
    g, p = gt.astype(np.int64).ravel(), pred.astype(np.int64).ravel()
    m = (g > 0) & (p > 0)
    ng, npd = int(gt.max()) + 1, int(pred.max()) + 1
    inter = np.bincount(g[m] * npd + p[m], minlength=ng * npd).reshape(ng, npd)
    ag, ap = np.bincount(g, minlength=ng), np.bincount(p, minlength=npd)
    gi, pi = np.nonzero(inter)
    return {(int(a), int(b)): inter[a, b] / (ag[a] + ap[b] - inter[a, b]) for a, b in zip(gi, pi) if a > 0 and b > 0}


def matched_pred_ids(gt, pred, thr=0.5):
    """ids of predicted instances matched (IoU > thr) to some gt instance; IoU > 0.5 makes the match unique."""
    return {p for (g, p), v in iou_pairs(gt, pred).items() if v > thr}


def fp_audit(pred, own, sibs):
    """returns dict(fp, fp_in_sibling, tp): predicted ids not matched in `own`, how many of them match an instance in any of `sibs` (list of label images)."""
    ids = [i for i in np.unique(pred) if i > 0]
    tp = matched_pred_ids(own, pred) if own.max() > 0 else set()
    fp = [i for i in ids if i not in tp]
    other = set()
    for s in sibs:
        if s.max() > 0:
            other |= matched_pred_ids(s, pred)
    return {"pred": len(ids), "tp": len(tp), "fp": len(fp), "fp_in_sibling": sum(1 for i in fp if i in other)}

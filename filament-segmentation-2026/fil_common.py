"""Shared helpers for the filament instance-segmentation task: records, grouped split, rasterisation, preprocessing, panoptic quality, RLE.

Everything here is CPU/numpy code with no dependence on the training framework, so the tests and the notebook import the same functions.
Conventions: images are 2048 x 2048 uint8 JPEGs; a 'record' is one annotator's set of polygons for one file (the task says to treat the
copies of one file as different images); the split is by FILE so that every copy of a file lands in the same role.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

S = 2048
ROLES = ("fit", "choose", "report")


# ---------------------------------------------------------------------------------------------------------------- records and split
def find_root(root=None):
    """data root holding train/ and test/; FIL_ROOT overrides; prints nothing (callers print where they ran)."""
    root = root or os.environ.get("FIL_ROOT") or "data/MAGFiLO_1.0_Kaggle_2026"
    assert os.path.isdir(os.path.join(root, "train")), f"no train folder under {root}"
    return root


def load_records(root=None):
    """list of dicts {rid, file, annotator, polys (list of flat [x0,y0,x1,y1,...]), cats} in a fixed order (sorted by file then id)."""
    root = find_root(root)
    jp = next(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(root, "train")) for f in fs if f.endswith(".json"))
    d = json.load(open(jp))
    by = {}
    for a in d["annotations"]:
        if a["segmentation"]:
            by.setdefault(a["image_id"], []).append(a)
    recs = []
    for im in d["images"]:
        anns = by.get(im["id"], [])
        recs.append({"rid": im["id"], "file": im["file_name"], "annotator": im["id"].split("-")[0],
                     "polys": [a["segmentation"][0] for a in anns], "cats": [a["category_id"] for a in anns]})
    recs.sort(key=lambda r: (r["file"], r["rid"]))
    return recs


def instrument(fname):
    """two-letter instrument/station code at the end of the file stem (e.g. '20140609195854Bh.jpeg' -> 'Bh')."""
    return os.path.splitext(fname)[0][-2:]


def split_files(files, seed=0, fracs=(0.70, 0.15, 0.15)):
    """role -> sorted list of FILES, stratified by instrument code, deterministic in the seed. Copies of one file always share a role."""
    assert abs(sum(fracs) - 1) < 1e-9
    rng = np.random.RandomState(seed)
    out = {r: [] for r in ROLES}
    groups = {}
    for f in sorted(set(files)):
        groups.setdefault(instrument(f), []).append(f)
    for g in sorted(groups):
        fs = list(groups[g])
        rng.shuffle(fs)
        n = len(fs)
        a, b = int(round(fracs[0] * n)), int(round((fracs[0] + fracs[1]) * n))
        out["fit"] += fs[:a]; out["choose"] += fs[a:b]; out["report"] += fs[b:]
    return {r: sorted(v) for r, v in out.items()}


def role_of_records(recs, split):
    which = {f: r for r, fs in split.items() for f in fs}
    return {r["rid"]: which[r["file"]] for r in recs}


# ---------------------------------------------------------------------------------------------------------------- rasterisation
def rasterize(polys, size=S):
    """int32 label image (0 background, 1..n instances); later polygons overwrite earlier ones where they overlap."""
    img = Image.new("I", (size, size), 0)
    dr = ImageDraw.Draw(img)
    for i, p in enumerate(polys, 1):
        dr.polygon([float(v) for v in p], fill=i, outline=i)
    return np.asarray(img, dtype=np.int32)


# ---------------------------------------------------------------------------------------------------------------- preprocessing
def disk_params(img):
    """(cy, cx, r) of the solar disk from a uint8 image: threshold, fill, largest component; r from the area."""
    thr = max(8.0, 0.25 * float(np.percentile(img, 99)))
    m = img > thr
    m = ndi.binary_opening(m, structure=np.ones((5, 5)))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return S / 2.0, S / 2.0, S / 2.0
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    comp = lab == k
    cy, cx = ndi.center_of_mass(comp)
    return float(cy), float(cx), float(np.sqrt(comp.sum() / np.pi))


def _smooth(a, sigma, f=4):
    """Gaussian smoothing of sigma pixels computed at 1/f resolution (block mean, sigma/f, bilinear back): same smoothing scale, ~f^2 cheaper."""
    h, w = a.shape
    lo = a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    lo = ndi.gaussian_filter(lo, sigma / f, mode="nearest")
    return ndi.zoom(lo, f, order=1, mode="nearest").astype(np.float32)


def preprocess(img, mode="flat_lcn"):
    """uint8 (S,S) -> uint8 (S,S) quantised model input, 0 = -4 sigma ... 255 = +4 sigma; off-disk pixels are the mid value (0 after centring).
    mode 'raw': per-image standardisation on the disk. mode 'flat_lcn': radial flattening of limb darkening, then local contrast normalisation.
    Constants (radial bins, 20 px Gaussian) are fixed here, not fitted on any role; the paper that suggests this style of pipeline does not ablate it."""
    x = img.astype(np.float32)
    cy, cx, r = disk_params(img)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rho = np.hypot(yy - cy, xx - cx) / max(r, 1.0)
    disk = rho < 0.985
    if mode == "flat_lcn":
        b = np.clip((rho * 100).astype(np.int32), 0, 119)
        cnt = np.bincount(b[disk], minlength=120).astype(np.float64)
        s = np.bincount(b[disk], weights=x[disk], minlength=120)
        prof = np.where(cnt > 0, s / np.maximum(cnt, 1), 1.0)
        prof = ndi.gaussian_filter1d(prof, 1.5)
        x = x / np.maximum(prof[b].astype(np.float32), 1.0)
        w = _smooth(disk.astype(np.float32), 20)
        g = _smooth(x * disk, 20) / np.maximum(w, 1e-3)
        v = _smooth(((x - g) ** 2) * disk, 20) / np.maximum(w, 1e-3)
        z = (x - g) / np.sqrt(v + 1e-4)
    elif mode == "raw":
        mu, sd = x[disk].mean(), x[disk].std() + 1e-6
        z = (x - mu) / sd
    else:
        raise ValueError(mode)
    z = np.where(disk, z, 0.0)
    return np.clip((z + 4.0) * 31.875, 0, 255).astype(np.uint8)


def to_float(xq):
    """inverse of the quantisation: uint8 -> float32 roughly in [-4, 4]."""
    return xq.astype(np.float32) / 31.875 - 4.0


# ---------------------------------------------------------------------------------------------------------------- panoptic quality
def pq(gt, pred, thr=0.5):
    """panoptic quality of label image `pred` against `gt` (0 = background, labels 1..n).  match = IoU > thr (unique for thr >= 0.5).
    returns pq, sq, rq, tp, fp, fn and the list of matched IoUs."""
    na, nb = int(gt.max()), int(pred.max())
    if na == 0 and nb == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0, "ious": [], "gt_iou": [], "gt_area": [], "pred_hit": []}
    m = (gt > 0) & (pred > 0)
    inter = np.bincount(gt[m].astype(np.int64) * (nb + 1) + pred[m], minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    area_a = np.bincount(gt.ravel(), minlength=na + 1)
    area_b = np.bincount(pred.ravel(), minlength=nb + 1)
    ious = []
    gt_iou = np.zeros(na, dtype=np.float32)          # IoU of the matched prediction per ground-truth instance (0 = unmatched)
    pred_hit = np.zeros(nb, dtype=bool)
    for i in range(1, na + 1):
        for j in np.nonzero(inter[i, 1:])[0] + 1:
            u = area_a[i] + area_b[j] - inter[i, j]
            v = inter[i, j] / u if u else 0.0
            if v > thr:
                ious.append(float(v)); gt_iou[i - 1] = v; pred_hit[j - 1] = True
    tp = len(ious)
    fp, fn = nb - tp, na - tp
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": (sum(ious) / den) if den else 1.0, "sq": float(np.mean(ious)) if ious else 0.0, "rq": tp / den if den else 1.0,
            "tp": tp, "fp": fp, "fn": fn, "ious": ious, "gt_iou": gt_iou.tolist(), "gt_area": area_a[1:].tolist(), "pred_hit": pred_hit.tolist()}


def relations(gt, pred, min_frac=0.1):
    """one-to-many and many-to-one counts: ground-truth instances that overlap >= 2 predictions (split), predictions that overlap >= 2
    ground-truth instances (merge); an overlap counts when it covers at least `min_frac` of the smaller of the two regions."""
    na, nb = int(gt.max()), int(pred.max())
    if na == 0 or nb == 0:
        return {"gt_split": 0, "pred_merge": 0, "gt": na, "pred": nb}
    m = (gt > 0) & (pred > 0)
    inter = np.bincount(gt[m].astype(np.int64) * (nb + 1) + pred[m], minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)[1:, 1:]
    area_a = np.bincount(gt.ravel(), minlength=na + 1)[1:][:, None]
    area_b = np.bincount(pred.ravel(), minlength=nb + 1)[1:][None, :]
    ov = inter >= min_frac * np.minimum(area_a, area_b)
    return {"gt_split": int((ov.sum(1) >= 2).sum()), "pred_merge": int((ov.sum(0) >= 2).sum()), "gt": na, "pred": nb}


def dice_iou_pixel(gt, pred):
    g, p = gt > 0, pred > 0
    inter = float((g & p).sum())
    return {"dice": 2 * inter / max(g.sum() + p.sum(), 1), "iou": inter / max((g | p).sum(), 1)}


# ---------------------------------------------------------------------------------------------------------------- instances and RLE
def instances_from_prob(prob, thr=0.5, min_area=30, close_radius=0):
    """probability map -> instance label image: threshold, optional binary closing (disk of `close_radius`), 8-connected components,
    components below `min_area` pixels dropped. Parameters are chosen on the CHOOSE role only."""
    m = prob > thr
    if close_radius > 0:
        k = int(close_radius)
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        m = ndi.binary_closing(m, structure=(yy ** 2 + xx ** 2) <= k * k)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return lab.astype(np.int32)
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    keep[0] = False
    remap = np.cumsum(keep) * keep
    return remap[lab].astype(np.int32)


def rle_counts(mask):
    """boolean/uint8 (S,S) mask -> the COCO compressed-RLE counts string the platform asks for (size is fixed to 2048 x 2048)."""
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    c = r["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def rle_decode(counts, size=S):
    from pycocotools import mask as mu
    return mu.decode({"size": [size, size], "counts": counts.encode("ascii") if isinstance(counts, str) else counts})


def submission_rows(file_stem, inst_labels):
    """rows (filament_id, counts) for one image: ids '<stem>_<n>' with n = 1..number of instances."""
    n = int(inst_labels.max())
    return [(f"{file_stem}_{i}", rle_counts(inst_labels == i)) for i in range(1, n + 1)]

"""Phase-1 data investigation for the filament instance-segmentation task. Runs on the local CPU, reads only the provided files.
usage: python3 helpers/investigate_data.py [--root data/MAGFiLO_1.0_Kaggle_2026] [--out traces/data_investigation.json]
Outputs one JSON of numbers (no figures): sizes and copies, annotator codes, instrument/time shift, instance counts and areas, widths, extents,
and the agreement between annotator copies of the same image measured with the competition metric (panoptic quality at IoU > 0.5)."""
import argparse
import collections
import json
import multiprocessing as mp
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/MAGFiLO_1.0_Kaggle_2026")
ap.add_argument("--out", default="traces/data_investigation.json")
ap.add_argument("--max-pairs", type=int, default=400)
ap.add_argument("--n-masks", type=int, default=250)
ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
args = ap.parse_args()
S = 2048


def pct(a, qs=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    a = np.asarray(a, dtype=float)
    return {str(q): float(np.percentile(a, q)) for q in qs} if len(a) else {}


def raster_instances(polys):
    """label image (0 = background, 1..n = instances); later instances overwrite earlier ones where they overlap."""
    img = Image.new("I", (S, S), 0)
    dr = ImageDraw.Draw(img)
    for i, poly in enumerate(polys, 1):
        dr.polygon([float(v) for v in poly], fill=i, outline=i)
    return np.asarray(img, dtype=np.int32)


def pq(a, b):
    """panoptic quality of label image b against a (both: 0 background), match at IoU > 0.5; also SQ, RQ."""
    na, nb = a.max(), b.max()
    if na == 0 and nb == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    m = (a > 0) & (b > 0)
    inter = np.bincount(a[m].astype(np.int64) * (nb + 1) + b[m], minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    area_a = np.bincount(a.ravel(), minlength=na + 1)
    area_b = np.bincount(b.ravel(), minlength=nb + 1)
    tp, ious = 0, []
    for i in range(1, na + 1):
        for j in np.nonzero(inter[i, 1:])[0] + 1:
            u = area_a[i] + area_b[j] - inter[i, j]
            iou = inter[i, j] / u if u else 0.0
            if iou > 0.5:
                tp += 1
                ious.append(iou)
    fp, fn = nb - tp, na - tp
    den = tp + 0.5 * fp + 0.5 * fn
    pqv = (sum(ious) / den) if den else 1.0
    return {"pq": pqv, "sq": float(np.mean(ious)) if ious else 0.0, "rq": tp / den if den else 1.0, "tp": tp, "fp": fp, "fn": fn}


d = json.load(open(next(os.path.join(r, f) for r, _, fs in os.walk(args.root) for f in fs if f.endswith(".json"))))
res = {}
imgs = {im["id"]: im for im in d["images"]}
anns = collections.defaultdict(list)
for a in d["annotations"]:
    anns[a["image_id"]].append(a)
files = collections.defaultdict(list)
for iid, im in imgs.items():
    files[im["file_name"]].append(iid)
train_dir = os.path.join(args.root, "train", "train_images")
test_dir = os.path.join(args.root, "test", "test_images")
train_files = sorted(os.listdir(train_dir))
test_files = sorted(os.listdir(test_dir))
res["counts"] = {"image_records": len(imgs), "distinct_files_in_json": len(files), "train_jpegs": len(train_files), "test_jpegs": len(test_files),
                 "annotations": len(d["annotations"]), "json_files_not_in_train_dir": len(set(files) - set(train_files)),
                 "train_files_without_record": len(set(train_files) - set(files)), "overlap_train_test_filenames": len(set(train_files) & set(test_files)),
                 "records_with_zero_annotations": sum(1 for i in imgs if not anns[i])}
res["copies_per_file"] = dict(collections.Counter(len(v) for v in files.values()))
prefix = collections.Counter(i.split("-")[0] for i in imgs)
res["annotator_prefix_counts"] = dict(prefix.most_common(30))
res["n_distinct_prefix"] = len(prefix)
res["category_counts"] = dict(collections.Counter(a["category_id"] for a in d["annotations"]))
res["segmentation_parts"] = dict(collections.Counter(len(a["segmentation"]) for a in d["annotations"]))
res["iscrowd"] = dict(collections.Counter(a.get("iscrowd", 0) for a in d["annotations"]))
res["annotations_per_record"] = pct([len(anns[i]) for i in imgs])
res["annotation_area"] = pct([a["area"] for a in d["annotations"]])
res["bbox_max_side"] = pct([max(a["bbox"][2], a["bbox"][3]) for a in d["annotations"]])
res["bbox_min_side"] = pct([min(a["bbox"][2], a["bbox"][3]) for a in d["annotations"]])
res["polygon_vertices"] = pct([len(a["segmentation"][0]) // 2 for a in d["annotations"] if a["segmentation"]])


def ym(fn):
    return fn[:6]


def instr(fn):
    return fn[14:16]


res["shift"] = {
    "train_year_counts": dict(sorted(collections.Counter(f[:4] for f in train_files).items())),
    "test_year_counts": dict(sorted(collections.Counter(f[:4] for f in test_files).items())),
    "train_instrument_counts": dict(collections.Counter(instr(f) for f in train_files)),
    "test_instrument_counts": dict(collections.Counter(instr(f) for f in test_files)),
    "train_date_range": [min(f[:8] for f in train_files), max(f[:8] for f in train_files)],
    "test_date_range": [min(f[:8] for f in test_files), max(f[:8] for f in test_files)],
    "test_months_also_in_train": sum(1 for m in {ym(f) for f in test_files} if m in {ym(f) for f in train_files}),
    "test_distinct_months": len({ym(f) for f in test_files}),
}

# mask-based statistics on a sample of record copies
rng = np.random.RandomState(0)
all_ids = list(imgs)
sample = rng.choice(len(all_ids), size=min(args.n_masks, len(all_ids)), replace=False)


def mask_one(k):
    """per-record mask statistics (runs in a worker process; reads the module-level annotation tables inherited by fork)."""
    iid = all_ids[k]
    polys = [a["segmentation"][0] for a in anns[iid] if a["segmentation"]]
    if not polys:
        return None
    lab = raster_instances(polys)
    fg = lab > 0
    edt = ndi.distance_transform_edt(fg)
    ridge = (edt == ndi.maximum_filter(edt, size=3)) & fg
    w = 2.0 * edt[ridge]
    cc, ncc = ndi.label(fg, structure=np.ones((3, 3)))
    return {"cov": float(fg.mean()), "n": len(polys), "wmed": float(np.median(w)) if len(w) else None,
            "wp10": float(np.percentile(w, 10)) if len(w) else None, "ncc": int(ncc)}


def pair_one(ab):
    a_id, b_id = ab
    pa = [x["segmentation"][0] for x in anns[a_id] if x["segmentation"]]
    pb = [x["segmentation"][0] for x in anns[b_id] if x["segmentation"]]
    if not pa and not pb:
        return None
    la, lb = raster_instances(pa), raster_instances(pb)
    r = pq(la, lb)
    fa, fb = la > 0, lb > 0
    r["iou_sem"] = float((fa & fb).sum() / max((fa | fb).sum(), 1))
    r["dn"] = len(pa) - len(pb)
    return r


pool = mp.get_context("fork").Pool(args.workers)
rows = [r for r in pool.map(mask_one, [int(k) for k in sample], chunksize=4) if r]
coverage = [r["cov"] for r in rows]
n_inst = [r["n"] for r in rows]
width_med = [r["wmed"] for r in rows if r["wmed"] is not None]
width_p10 = [r["wp10"] for r in rows if r["wp10"] is not None]
frag = [(r["n"], r["ncc"]) for r in rows]
res["mask_stats_sample"] = {
    "n_records_sampled": len(coverage),
    "foreground_fraction": pct(coverage),
    "instances_per_record": pct(n_inst),
    "per_record_median_width_px": pct(width_med),
    "per_record_p10_width_px": pct(width_p10),
    "instances_vs_connected_components": {"mean_instances": float(np.mean([a for a, _ in frag])), "mean_components_8conn": float(np.mean([b for _, b in frag])),
                                         "share_records_with_touching_instances": float(np.mean([b < a for a, b in frag]))},
}

# agreement between annotator copies of the same file, scored with the competition metric
pairs = [(v[i], v[j]) for v in files.values() if len(v) > 1 for i in range(len(v)) for j in range(i + 1, len(v))]
rng.shuffle(pairs)
prs = [r for r in pool.map(pair_one, pairs[:args.max_pairs], chunksize=4) if r]
pool.close(); pool.join()
pq_l = [r["pq"] for r in prs]; sq_l = [r["sq"] for r in prs]; rq_l = [r["rq"] for r in prs]
iou_sem = [r["iou_sem"] for r in prs]; dn = [r["dn"] for r in prs]
res["annotator_agreement"] = {"n_pairs_scored": len(pq_l), "n_pairs_total": len(pairs), "pq_between_copies": pct(pq_l), "sq": pct(sq_l), "rq": pct(rq_l),
                              "semantic_iou": pct(iou_sem), "instance_count_difference": pct(dn)}
# record-level details on which copies exist
res["records_per_file_examples"] = {k: v for k, v in list(files.items())[:3]}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("counts", "copies_per_file", "n_distinct_prefix", "category_counts", "segmentation_parts", "iscrowd")}, indent=1))
print("wrote", args.out)

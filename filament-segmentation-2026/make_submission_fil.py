"""Write the submission CSV from a trained run (one row per predicted filament: filament_id '<stem>_<n>', segmentation_rle = COCO compressed counts).
usage: python3 helpers/make_submission_fil.py --run runs/v0_s0 --prep prep --out submission.csv [--flip-tta]
Instance parameters come from the CHOOSE tuning stored in <run>/eval.json (frozen); no REPORT or test annotation is touched. Self-checks before writing:
unique ids, every counts string decodes to a non-empty 2048 x 2048 mask, ids carry the file stem, no quotation marks in the counts."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402
import fil_infer as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--prep", default="prep")
ap.add_argument("--mode", default="flat_lcn")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--flip-tta", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta, X, LAB, fidx = I.load_prep(args.prep, args.mode)
model, cfg = I.load_run(args.run, dev)
par = json.load(open(os.path.join(args.run, "eval.json")))["choose"]["params"]
print(f"[submit] run {args.run}, frozen instance parameters {par}, device {dev}", flush=True)
rows, ids, n_files = [], set(), 0
for (part, f), i in sorted(fidx.items()):
    if part != "test":
        continue
    prob = I.predict_prob(model, X[i], dev, args.flip_tta, **I.tile_args(cfg)).astype(np.float32)
    inst = F.instances_from_prob(prob, par["thr"], par["min_area"], par["close"])
    stem = os.path.splitext(f)[0]
    for fid, counts in F.submission_rows(stem, inst):
        assert fid not in ids and '"' not in counts and "'" not in counts
        ids.add(fid); rows.append((fid, counts))
    n_files += 1
for fid, counts in rows[:50]:
    m = F.rle_decode(counts)
    assert m.shape == (F.S, F.S) and m.sum() > 0
with open(args.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["filament_id", "segmentation_rle"]); w.writerows(rows)
print(f"[submit] {len(rows)} filaments in {n_files} test images -> {args.out}", flush=True)

"""Write the Kaggle submission CSV (one row per predicted filament: filament_id, segmentation_rle) for the 180 TEST images from a frozen configuration:
runs (probability maps averaged) + instance-recovery parameters chosen on CHOOSE. Inference only; REPORT and labels are not touched.
usage: python3 submit_best.py --runs runs/v0_s0 ... --prep prep --params '{...}' --out submission.csv [--tta] [--limit N]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402
import fil_infer as I  # noqa: E402
import fil_post as Q  # noqa: E402
import fil_pre as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--prep", required=True)
ap.add_argument("--mode", default="flat_lcn")
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tta", action="store_true")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta, X, LAB, fidx = I.load_prep(args.prep, args.mode)
test_files = [f["file"] for f in meta["files"] if f["part"] == "test"]
if args.limit:
    test_files = test_files[:args.limit]
params = json.loads(args.params)
t0 = time.time()
maps_all = [[] for _ in test_files]                      # one model at a time on the device (the card is shared with training jobs): maps are kept as fp16 on the host
for run in args.runs:
    cfg = json.load(open(os.path.join(run, "config.json")))
    m, cfg = (I.load_run(run, dev) if cfg["variant"] in ("v0", "v1", "v2", "v3") else P.load_run2(run, dev))
    ta = I.tile_args(cfg)
    for i, f in enumerate(test_files):
        maps_all[i].append(np.asarray(I.predict_prob(m, X[fidx[("test", f)]], dev, args.tta, **ta), dtype=np.float16))
        if (i + 1) % 30 == 0 or i + 1 == len(test_files):
            print(f"[submit] {os.path.basename(run)} {i + 1}/{len(test_files)} test images, elapsed {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()
rows = []
for i, f in enumerate(test_files):
    prob = Q.ensemble(maps_all[i]) if len(maps_all[i]) > 1 else maps_all[i][0]
    rows += F.submission_rows(os.path.splitext(f)[0], Q.instances_post(prob, **params))
print(f"[submit] {len(rows)} filaments, elapsed {time.time() - t0:.0f}s", flush=True)
with open(args.out, "w") as fh:
    fh.write("filament_id,segmentation_rle\n")
    for k, c in rows:
        fh.write(f"{k},{c}\n")
print(f"[submit] wrote {args.out}: {len(rows)} rows for {len(test_files)} images; images without any filament: "
      f"{len(test_files) - len({k.rsplit('_', 1)[0] for k, _ in rows})}", flush=True)

"""Stage A, step 1 (GPU, inference only): dump the probability maps of CHOOSE files for a list of trained runs, so every later analysis (ensembles, instance-parameter
grids, false-positive audit) is CPU work on files.
usage: python3 predict_choose.py --runs runs/v0_s0 runs/v2_s0 ... --prep prep --out probs [--tta] [--native]
writes <out>/<run_name>.npy float16 (n_choose_files, S, S) in the order of <out>/files.json, and (--native, Mask2Former runs) <out>/<run_name>_native.npy uint8 instance labels.
Prints one line per run with the elapsed time (live)."""
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
ap.add_argument("--prep", default="prep")
ap.add_argument("--mode", default="flat_lcn")
ap.add_argument("--out", default="probs")
ap.add_argument("--role", default="choose", choices=["choose"], help="REPORT is read only by fil_infer.py --report, once per frozen run")
ap.add_argument("--tta", action="store_true", help="average with the horizontally flipped input")
ap.add_argument("--native", action="store_true", help="also the native Mask2Former instance readout (stitched), for m2fx / v3 runs")
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N CHOOSE files")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta, X, LAB, fidx = I.load_prep(args.prep, args.mode)
files = list(meta["split"]["choose"])
if args.limit:
    files = files[:args.limit]
os.makedirs(args.out, exist_ok=True)
json.dump(files, open(os.path.join(args.out, "files.json"), "w"))
for run in args.runs:
    name = os.path.basename(run.rstrip("/"))
    t0 = time.time()
    cfg = json.load(open(os.path.join(run, "config.json")))
    if cfg["variant"] in ("v0", "v1", "v2", "v3"):
        model, cfg = I.load_run(run, dev)
    else:
        model, cfg = P.load_run2(run, dev)
    ta = I.tile_args(cfg)
    is_m2f = cfg["variant"] in ("v3", "m2fx")
    out = np.lib.format.open_memmap(os.path.join(args.out, f"{name}.npy"), mode="w+", dtype=np.float16, shape=(len(files), F.S, F.S))
    nat = np.lib.format.open_memmap(os.path.join(args.out, f"{name}_native.npy"), mode="w+", dtype=np.uint8, shape=(len(files), F.S, F.S)) if (args.native and is_m2f) else None
    for i, f in enumerate(files):
        xq = X[fidx[("train", f)]]
        if nat is not None:
            pr, lab = Q.native_from_m2f(model, xq, dev, tile=ta["tile"], stride=ta["stride"], batch=ta["batch"])
            out[i] = pr; nat[i] = np.minimum(lab, 255).astype(np.uint8)
        else:
            out[i] = I.predict_prob(model, xq, dev, args.tta, **ta)
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[predict] {name} {i + 1}/{len(files)} files elapsed {time.time() - t0:.0f}s", flush=True)
    out.flush()
    if nat is not None:
        nat.flush()
    del model
    torch.cuda.empty_cache()
    print(f"[predict] {name} done in {time.time() - t0:.0f}s tta={args.tta}", flush=True)

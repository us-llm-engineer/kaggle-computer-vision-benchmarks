"""Score candidate probability sources on the CHOOSE records with ONE fixed instance-recovery setting (no search): used to pick the model/ensemble to submit.
usage: python3 score_fixed.py --prep prep --files probs/files.json --params '{"hi":0.4,"lo":0.2,"min_conf":0.8,"min_area":200,"close_radius":0}' \
         --src ens_v0=probs/v0_s0.npy,probs/v0_s1.npy,probs/v0_s2.npy --src r34e=probs/r34e_s0.npy [--max-records 80]
Each --src is name=<npy>[,<npy>...] (maps averaged); every map has the CHOOSE files in the order of --files. REPORT is never read."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402
import fil_infer as I  # noqa: E402
import fil_post as Q  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--files", required=True)
ap.add_argument("--params", required=True)
ap.add_argument("--src", action="append", required=True)
ap.add_argument("--max-records", type=int, default=80)
ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
args = ap.parse_args()
meta, X, LAB, fidx = I.load_prep(args.prep)
files = json.load(open(args.files))
frow = {f: i for i, f in enumerate(files)}
choose = [j for j, r in enumerate(meta["records"]) if r["role"] == "choose" and r["file"] in frow][:args.max_records]
params = json.loads(args.params)
SRC = {}
for s in args.src:
    n, paths = s.split("=")
    SRC[n] = [np.load(p, mmap_mode="r") for p in paths.split(",")]


def one(a):
    n, j = a
    f = meta["records"][j]["file"]
    maps = SRC[n]
    p = Q.ensemble([m[frow[f]] for m in maps]) if len(maps) > 1 else np.asarray(maps[0][frow[f]])
    r = F.pq(np.asarray(LAB[j]).astype(np.int32), Q.instances_post(p, **params))
    return n, r["pq"], r["sq"], r["rq"], r["tp"], r["fp"], r["fn"]


t0 = time.time()
with mp.get_context("fork").Pool(args.workers) as pool:
    for n in SRC:
        rows = pool.map(one, [(n, j) for j in choose])
        a = np.array([r[1:] for r in rows], float)
        print(f"[score] {n:14s} CHOOSE({len(choose)}) PQ {a[:, 0].mean():.4f} sq {a[:, 1].mean():.3f} rq {a[:, 2].mean():.3f} tp/fp/fn {int(a[:, 3].sum())}/{int(a[:, 4].sum())}/{int(a[:, 5].sum())} ({time.time() - t0:.0f}s)", flush=True)

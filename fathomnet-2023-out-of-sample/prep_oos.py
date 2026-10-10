"""Out-of-sample round, step 0: the split with three separate held-back category sets and the dataset statistics that depend on it.

The torchvision-v2 pre-processing output of the first attempt (uint8 arrays prep/train_x224.npy ...) is reused as it is; only the split, the channel
statistics of the new training set and the group sizes are produced here. Asserts that the groups are disjoint and that no held-out category reaches
the in-distribution pool.
usage: python3 prep_oos.py [--seed 0] [--n-select 4] [--n-pool 4]"""
import argparse
import json
import os

import numpy as np

import fn_common as C
import fn_oos_common as O

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-select", type=int, default=4); ap.add_argument("--n-pool", type=int, default=4)
args = ap.parse_args()
Y = np.load(f"{C.PREP}/train_y.npy")
sp = O.make_oos_split(Y, seed=args.seed, n_select=args.n_select, n_pool=args.n_pool)
held = sp["report_classes"] + sp["select_classes"] + sp["pool_classes"]
sets = {k: set(sp[k].tolist()) for k in ("train", "val_in", "report", "select", "pool")}
names = list(sets)
for i, a in enumerate(names):
    for b in names[i + 1:]: assert not (sets[a] & sets[b]), f"{a} and {b} overlap"
assert Y[sp["train"]][:, held].sum() == 0 and Y[sp["val_in"]][:, held].sum() == 0, "a held-out category reached the in-distribution pool"
assert len(set(sp["report_classes"]) & set(sp["select_classes"])) == 0 and len(set(sp["report_classes"]) & set(sp["pool_classes"])) == 0
assert (Y[sp["report"]][:, sp["report_classes"]].sum(1) > 0).all() and (Y[sp["select"]][:, sp["select_classes"]].sum(1) > 0).all()
seen = O.seen_columns(Y, sp["train"])
meta = {"seed": args.seed, "sizes": {k: int(len(sp[k])) for k in names}, "report_classes": [c + 1 for c in sp["report_classes"]],
        "select_classes": [c + 1 for c in sp["select_classes"]], "pool_classes": [c + 1 for c in sp["pool_classes"]],
        "seen_categories_in_train": int(len(seen)), "mixed_share": {g: float(O.mixed_mask(Y, sp[g], seen).mean()) for g in ("report", "select", "pool")}}
x_path = f"{C.PREP}/train_x224.npy"
if os.path.exists(x_path):                       # channel statistics from the NEW training set only
    X = np.load(x_path, mmap_mode="r"); tot = np.zeros(3); tot2 = np.zeros(3); n = 0
    for a in range(0, len(sp["train"]), 256):
        c = np.asarray(X[sp["train"][a:a + 256]]).astype(np.float64) / 255.0; tot += c.sum((0, 1, 2)); tot2 += (c ** 2).sum((0, 1, 2)); n += c.shape[0] * c.shape[1] * c.shape[2]
    mean = tot / n; std = np.sqrt(tot2 / n - mean ** 2); meta["channel_mean"] = mean.tolist(); meta["channel_std"] = std.tolist()
else:
    old = json.load(open(f"{C.PREP}/meta.json")); meta["channel_mean"], meta["channel_std"] = old["channel_mean"], old["channel_std"]; meta["channel_stats_from"] = "first attempt (arrays absent)"
np.savez(f"{C.PREP}/split_oos.npz", **{k: sp[k] for k in names}, report_classes=np.array(sp["report_classes"]), select_classes=np.array(sp["select_classes"]),
         pool_classes=np.array(sp["pool_classes"]))
json.dump(meta, open(f"{C.PREP}/meta_oos.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("sizes", "report_classes", "select_classes", "pool_classes", "seen_categories_in_train", "mixed_share")}, indent=1))

"""Preprocessing stage, run once before any training. Writes, under --out:
  x_<mode>.npy   uint8 (N, 2048, 2048) preprocessed inputs for all train files then all test files (memmap)
  lab.npy        uint8 (R, 2048, 2048) instance label image of every annotation record (0 = background)
  meta.json      files, records (index, role, instance count), split, disk parameters, dataset statistics, counts of repaired/dropped items
usage: python3 helpers/prep_fil.py --root data/MAGFiLO_1.0_Kaggle_2026 --out prep --mode flat_lcn --workers 4 [--limit N]
Every later run reads these arrays and the saved split, so all variants see identical inputs."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--out", default="prep")
ap.add_argument("--mode", default="flat_lcn", choices=["flat_lcn", "raw"])
ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto = cuda when available: batched nvjpeg decode and the transform on the GPU (the CPU path is the fallback and the reference)")
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N train files / records (keeps the split logic)")
args = ap.parse_args()

root = F.find_root(args.root)
os.makedirs(args.out, exist_ok=True)
recs = F.load_records(root)
train_files = sorted({r["file"] for r in recs})
test_files = sorted(os.listdir(os.path.join(root, "test", "test_images")))
if args.limit:
    train_files = train_files[:args.limit]; test_files = test_files[:max(2, args.limit // 4)]
    recs = [r for r in recs if r["file"] in set(train_files)]
all_files = [("train", f) for f in train_files] + [("test", f) for f in test_files]
split = F.split_files(train_files, args.seed)
role = F.role_of_records(recs, split)
N, R = len(all_files), len(recs)
print(f"[prep] root={root} out={args.out} mode={args.mode} files={N} (train {len(train_files)}, test {len(test_files)}) records={R}", flush=True)

xp = os.path.join(args.out, f"x_{args.mode}.npy")
lp = os.path.join(args.out, "lab.npy")
np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(N, F.S, F.S)).flush()
np.lib.format.open_memmap(lp, mode="w+", dtype=np.uint8, shape=(R, F.S, F.S)).flush()


def do_image(i):
    part, fname = all_files[i]
    img = np.asarray(Image.open(os.path.join(root, part, f"{part}_images", fname)).convert("L"))
    assert img.shape == (F.S, F.S), (fname, img.shape)
    cy, cx, r = F.disk_params(img)
    x = np.load(xp, mmap_mode="r+")
    x[i] = F.preprocess(img, args.mode)
    x.flush()
    return i, [cy, cx, r], float(img.mean())


def do_record(j):
    lab = F.rasterize(recs[j]["polys"])
    assert lab.max() < 256, (recs[j]["rid"], lab.max())
    l = np.load(lp, mmap_mode="r+")
    l[j] = lab.astype(np.uint8)
    l.flush()
    return j, int(lab.max()), float((lab > 0).mean())


t0 = time.time()
use_gpu = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
disk, repaired = {}, 0
if use_gpu:
    import fil_gpu as G
    dev = torch.device("cuda")
    xm = np.load(xp, mmap_mode="r+")
    for s0 in range(0, N, 16):
        idx = list(range(s0, min(N, s0 + 16)))
        raws = [open(os.path.join(root, all_files[i][0], f"{all_files[i][0]}_images", all_files[i][1]), "rb").read() for i in idx]
        try:
            imgs = G.decode_gray(raws, dev)
        except Exception:                                    # per-image CPU fallback (counted as repaired)
            imgs = []
            for i in idx:
                part, fname = all_files[i]
                imgs.append(torch.from_numpy(np.asarray(Image.open(os.path.join(root, part, f"{part}_images", fname)).convert("L")).copy()).to(dev)); repaired += 1
        for i, img in zip(idx, imgs):
            assert tuple(img.shape) == (F.S, F.S), (all_files[i][1], tuple(img.shape))
            out, (cy, cx, r) = G.preprocess_gpu(img, args.mode, return_disk=True)
            xm[i] = out.cpu().numpy()
            disk[i] = {"cy": cy, "cx": cx, "r": r, "mean": float(img.float().mean())}
        if (s0 // 16) % 5 == 0 or idx[-1] == N - 1:
            print(f"[prep] images {idx[-1] + 1}/{N} on cuda elapsed {time.time() - t0:.0f}s", flush=True)
    xm.flush()
with mp.get_context("fork").Pool(args.workers) as pool:
    if not use_gpu:
        for k, (i, d, m) in enumerate(pool.imap_unordered(do_image, range(N), chunksize=2), 1):
            disk[i] = {"cy": d[0], "cx": d[1], "r": d[2], "mean": m}
            if k % 50 == 0 or k == N:
                print(f"[prep] images {k}/{N} on cpu elapsed {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for k, (j, n, fg) in enumerate(pool.imap_unordered(do_record, range(R), chunksize=8), 1):
        rows[j] = (n, fg)
        if k % 200 == 0 or k == R:
            print(f"[prep] records {k}/{R} elapsed {time.time() - t0:.0f}s", flush=True)

radii = np.array([disk[i]["r"] for i in range(N)])
stats = {"roles": {r: {"files": len(split[r]), "records": sum(1 for v in role.values() if v == r),
                       "instances": int(sum(rows[j][0] for j in range(R) if role[recs[j]["rid"]] == r)),
                       "foreground_fraction_mean": float(np.mean([rows[j][1] for j in range(R) if role[recs[j]["rid"]] == r]))} for r in F.ROLES},
         "disk_radius_px": {"min": float(radii.min()), "median": float(np.median(radii)), "max": float(radii.max())},
         "dropped_or_repaired": {"images_decoded_by_cpu_fallback": repaired, "records_without_polygons": 0}, "device": "cuda" if use_gpu else "cpu",
         "elapsed_s": time.time() - t0}
meta = {"mode": args.mode, "seed": args.seed, "files": [{"part": p, "file": f} for p, f in all_files],
        "records": [{"rid": r["rid"], "file": r["file"], "annotator": r["annotator"], "role": role[r["rid"]], "n": rows[j][0]} for j, r in enumerate(recs)],
        "split": split, "disk": [disk[i] for i in range(N)], "stats": stats}
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
print("[prep] stats:", json.dumps(stats), flush=True)

"""Pre-processing stage, run once before any training.

  1. read the resized images and decode every one in batches on the GPU;
  2. validate: size, decode errors, ids missing from the labels or from the images (counts reported);
  3. store two uint8 versions of each image: 448x256 (full detail, for the tile model) and 224x128 (for the CNN and the classical features);
  4. align the multi-hot label matrix to the image order;
  5. make the train / in-distribution validation / novel validation split once, with a seed;
  6. compute per-channel mean and std on the TRAINING split only;
  7. write the dataset statistics (class balance, labels per image, brightness, split sizes) to prep/meta.json.

usage: python3 preprocess.py [--source <folder>] [--novel-range LO HI] [--n-novel N] [--limit N]"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms.v2 as T

import fn_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=C.DATA, help="folder holding img448_train and img448_eval, written by fetch_images.py"); ap.add_argument("--novel-range", type=int, nargs=2, default=[40, 150]); ap.add_argument("--n-novel", type=int, default=6)
ap.add_argument("--limit", type=int, default=0, help="stop after this many images per split (smoke tests)")
args = ap.parse_args()
os.makedirs(C.PREP, exist_ok=True); t0 = time.time()
W, H = 448, 256


DEV = os.environ.get("FN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# torchvision transforms v2 pipeline, applied to uint8 batches on the device: float conversion, bilinear (antialiased) resize, back to uint8
to_big = T.Compose([T.ToDtype(torch.float32, scale=True), T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.ToDtype(torch.uint8, scale=True)])
to_small = T.Compose([T.ToDtype(torch.float32, scale=True), T.Resize((H // 2, W // 2), interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.ToDtype(torch.uint8, scale=True)])
EXPECTED = {"train": 5950, "eval": 10744}


def decode_batch(items):
    """JPEG bytes -> (names, big uint8 (n,3,H,W), small uint8 (n,3,H/2,W/2), errors). Batched nvjpeg decode on the GPU, per-image CPU fallback."""
    from torchvision.io import ImageReadMode, decode_jpeg

    raws = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for _, b in items]; imgs, errors = None, []
    try:
        imgs = decode_jpeg(raws, device=DEV, mode=ImageReadMode.RGB)
    except Exception:
        imgs = []
        for (name, _), r in zip(items, raws):
            try: imgs.append(decode_jpeg(r, mode=ImageReadMode.RGB).to(DEV))
            except Exception as exc: imgs.append(None); errors.append((name, str(exc)))
    names = [n for (n, _), im in zip(items, imgs) if im is not None]; good = [im for im in imgs if im is not None]
    if not good: return names, None, None, errors
    same = all(tuple(im.shape[1:]) == (H, W) for im in good)
    big = torch.stack(good) if same else torch.stack([im if tuple(im.shape[1:]) == (H, W) else to_big(im[None])[0] for im in good])
    return names, big, to_small(big), errors


def read_split(split):
    """Decode a split straight into disk-backed arrays (low peak memory): returns ids, 448x256 array, 224x128 array, decode errors.
    Images are decoded and resized in batches on the device."""
    n_max = args.limit or EXPECTED[split]
    big = np.lib.format.open_memmap(f"{C.PREP}/{split}_x448.npy", mode="w+", dtype=np.uint8, shape=(n_max, H, W, 3))
    small = np.lib.format.open_memmap(f"{C.PREP}/{split}_x224.npy", mode="w+", dtype=np.uint8, shape=(n_max, H // 2, W // 2, 3))
    ids, bad, n = [], [], 0
    groups = iter([list(itertools.islice(C.iter_local_images(os.path.join(args.source, f"img448_{split}")), n_max))])
    for group in groups:
        for start in range(0, len(group), 256):
            if n >= n_max: break
            names, b, sm, errs = decode_batch(group[start:start + 256]); bad += errs
            if b is None: continue
            take = min(len(names), n_max - n)
            big[n:n + take] = b[:take].permute(0, 2, 3, 1).cpu().numpy(); small[n:n + take] = sm[:take].permute(0, 2, 3, 1).cpu().numpy()
            ids += names[:take]; n += take
        print(f"  {split}: {n} decoded and resized on {DEV} ({time.time() - t0:.0f}s)", flush=True)
        group.clear()
    big.flush(); small.flush()
    if n < n_max:   # fewer images than expected: store exactly n
        for arr, fname in ((big, "x448"), (small, "x224")):
            data = np.array(arr[:n]); del arr; np.save(f"{C.PREP}/{split}_{fname}.npy", data)
        big = np.load(f"{C.PREP}/{split}_x448.npy", mmap_mode="r"); small = np.load(f"{C.PREP}/{split}_x224.npy", mmap_mode="r")
    return np.array(ids), big, small, bad


report = {}
tr_ids, tr_big, tr_small, tr_bad = read_split("train")
ev_ids, ev_big, ev_small, ev_bad = read_split("eval")
report["decode_errors"] = {"train": len(tr_bad), "eval": len(ev_bad)}
label_ids, Y_all = C.load_labels(); pos = {i: k for k, i in enumerate(label_ids)}
keep = np.array([i in pos for i in tr_ids]); report["train_images_without_labels"] = int((~keep).sum())
report["labels_without_images"] = int(len(label_ids) - len(set(tr_ids) & set(label_ids)))
assert keep.all(), 'images without labels: rebuild the arrays before continuing'
Y = Y_all[[pos[i] for i in tr_ids]]
print("validation:", report, flush=True)

split = C.make_split(Y, seed=0, n_novel=args.n_novel, novel_range=tuple(args.novel_range))
tri = split["train"]; tot = np.zeros(3); tot2 = np.zeros(3); npx = 0   # channel statistics on the TRAINING split only, in chunks
for a in range(0, len(tri), 256):
    c = torch.from_numpy(np.asarray(tr_small[tri[a:a + 256]])).to(DEV).double() / 255.0; tot += c.sum((0, 1, 2)).cpu().numpy(); tot2 += (c ** 2).sum((0, 1, 2)).cpu().numpy(); npx += c.shape[0] * c.shape[1] * c.shape[2]
mean = tot / npx; std = np.sqrt(tot2 / npx - mean ** 2)
counts = Y.sum(0); used = counts > 0; per_img = Y.sum(1); gray = np.concatenate([np.asarray(tr_small[a:a + 512]).mean((1, 2, 3)) for a in range(0, len(tr_small), 512)])
meta = {
    "n_train_images": int(len(tr_ids)), "n_eval_images": int(len(ev_ids)), "size_big": [H, W], "size_small": [128, 224],
    "channel_mean": mean.tolist(), "channel_std": std.tolist(),
    "classes_with_any_label": int(used.sum()), "classes_never_seen": int((~used).sum()),
    "labels_per_image": {"mean": float(per_img.mean()), "max": int(per_img.max()), "share_single_label": float((per_img == 1).mean())},
    "most_common_class": {"category": int(counts.argmax() + 1), "images": int(counts.max())},
    "classes_with_at_most_5_images": int(((counts > 0) & (counts <= 5)).sum()),
    "brightness_quantiles_0_25_50_75_100": np.quantile(gray, [0, .25, .5, .75, 1]).round(1).tolist(),
    "split": {k: int(len(split[k])) for k in ("train", "val_in", "val_novel")}, "novel_classes": [c + 1 for c in split["novel_classes"]],
    "novel_class_image_counts": [int(counts[c]) for c in split["novel_classes"]], "validation_report": report, "seconds": round(time.time() - t0, 1),
}
np.save(f"{C.PREP}/train_ids.npy", tr_ids); np.save(f"{C.PREP}/eval_ids.npy", ev_ids); np.save(f"{C.PREP}/train_y.npy", Y)
np.savez(f"{C.PREP}/split.npz", train=split["train"], val_in=split["val_in"], val_novel=split["val_novel"], novel_classes=np.array(split["novel_classes"]))
json.dump(meta, open(f"{C.PREP}/meta.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("n_train_images", "n_eval_images", "channel_mean", "channel_std", "split", "novel_classes", "classes_with_any_label", "seconds")}, indent=1))

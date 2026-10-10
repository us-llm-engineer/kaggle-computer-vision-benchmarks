"""Shared pieces for the FathomNet 2023 experiments: label parsing, the two competition metrics, the validation split with its out-of-sample
proxy, the out-of-sample scores, and a reader for the resized images written by fetch_images.py.

Competition score = 0.5 * (sAUC + MAP@20), sAUC = 2 * AUC - 1 for the out-of-sample (osd) score. The test set has no labels, so a local
proxy is built from the training images only: a few mid-frequency categories are held out of training and every image that shows one of them
counts as "novel" in validation (see make_split)."""
import ast
import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FN_DATA", f"{ROOT}/data")
PREP = os.environ.get("FN_PREP", f"{ROOT}/prep")
OUT = os.environ.get("FN_OUT", f"{ROOT}/out")
CKPT = os.environ.get("FN_CKPT", f"{ROOT}/ckpt")
LOGS = os.environ.get("FN_LOGS", f"{ROOT}/logs")
NCLS = 290


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def load_labels(csv_path=None):
    """(ids, Y) with Y a (n, 290) multi-hot matrix; category k (1..290) is column k-1."""
    df = pd.read_csv(csv_path or f"{DATA}/multilabel_classification/train.csv")
    Y = np.zeros((len(df), NCLS), np.uint8)
    for i, cats in enumerate(df.categories):
        for c in ast.literal_eval(cats):
            Y[i, int(c) - 1] = 1
    return df.id.astype(str).values, Y


def category_key():
    """DataFrame with columns id, name, supercat (supercat is the 20-way semantic group)."""
    return pd.read_csv(f"{DATA}/category_key.csv")


def supercat_index():
    key = category_key(); names = sorted(key.supercat.unique()); lookup = {n: i for i, n in enumerate(names)}
    sc = np.full(NCLS, -1); sc[key.id.values.astype(int) - 1] = [lookup[s] for s in key.supercat]
    return sc, names


# ----------------------------------------------------------------------------------------------------------------- metrics
def map_at_k(scores, Y, k=20):
    """Mean average precision at k over images, for a (n, 290) score matrix and multi-hot truth. Every class appears once in a ranking, so
    repeated predictions of the same label (which the competition ignores) cannot occur. AP is normalised by min(#true labels, k)."""
    top = np.argsort(-scores, axis=1)[:, :k]
    rel = np.take_along_axis(Y, top, axis=1).astype(np.float64)
    precision = np.cumsum(rel, axis=1) / np.arange(1, k + 1)
    denominator = np.minimum(Y.sum(1), k).clip(min=1)
    return float(((precision * rel).sum(1) / denominator).mean())


def ap_list(predicted, truth, k=20):
    """Reference implementation on plain lists: once a correct label has scored, later repeats of it are skipped."""
    seen, hits, total = set(), 0, 0.0
    for rank, label in enumerate(predicted[:k], start=1):
        if label in truth and label not in seen:
            seen.add(label); hits += 1; total += hits / rank
    return total / max(min(len(truth), k), 1)


def sauc(ood_score, is_novel):
    """2 * AUC - 1, where a higher score should mean 'out of sample'."""
    return 2.0 * roc_auc_score(np.asarray(is_novel).astype(int), ood_score) - 1.0


def ood_scores(logits):
    """Three out-of-sample scores from multi-label logits (higher = more novel): 1 - max probability, negative energy, entropy."""
    logits = np.asarray(logits, np.float64); p = 1.0 / (1.0 + np.exp(-logits)); q = p / p.sum(1, keepdims=True).clip(min=1e-12)
    m = logits.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
    return {"msp": 1.0 - p.max(1), "energy": -lse, "entropy": -(q * np.log(q.clip(min=1e-12))).sum(1)}


def score_split(logits, Y, novel):
    """Metrics on the validation split: MAP@20 on in-distribution images, sAUC for each score, and the competition proxy."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))); inn = ~novel; out = {"map20": map_at_k(p[inn], Y[inn])}
    for name, s in ood_scores(logits).items():
        out[f"sauc_{name}"] = sauc(s, novel)
    out["top1_hit"] = float(Y[inn][np.arange(inn.sum()), p[inn].argmax(1)].mean())
    out["proxy_score"] = 0.5 * (out["map20"] + out["sauc_msp"])
    return out


# ------------------------------------------------------------------------------------------------------------------- split
def make_split(Y, seed=0, n_novel=6, novel_range=(40, 150), val_fraction=0.15):
    """Train / in-distribution validation / novel validation indices.

    Novel: n_novel categories of mid frequency are chosen at random and every image that shows any of them is removed from training and used as
    'novel' in validation. In-distribution validation is a random sample (stratified on each image's rarest label) of the remaining images."""
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed); counts = Y.sum(0)
    pool = [c for c in range(NCLS) if novel_range[0] <= counts[c] <= novel_range[1]]
    novel_classes = sorted(rng.choice(pool, size=min(n_novel, len(pool)), replace=False).tolist())
    is_novel_img = Y[:, novel_classes].sum(1) > 0; rest = np.where(~is_novel_img)[0]
    masked = np.where(Y[rest] > 0, counts[None, :], np.inf); rarest = masked.argmin(1)
    values, freq = np.unique(rarest, return_counts=True); rare_ok = set(values[freq >= 2].tolist())
    strata = np.array([c if c in rare_ok else -1 for c in rarest])
    train_idx, val_idx = train_test_split(rest, test_size=val_fraction, stratify=strata, random_state=seed)
    return {"train": np.sort(train_idx), "val_in": np.sort(val_idx), "val_novel": np.where(is_novel_img)[0], "novel_classes": novel_classes}


# ------------------------------------------------------------------------------------------------------------------ images
def iter_local_images(folder):
    for name in sorted(os.listdir(folder)):
        if name.endswith(".jpg"):
            with open(os.path.join(folder, name), "rb") as f:
                yield stem(name), f.read()


def load_prep():
    """Arrays written by preprocess.py."""
    meta = json.load(open(f"{PREP}/meta.json"))
    z = np.load(f"{PREP}/split.npz")
    return meta, {k: z[k] for k in z.files}

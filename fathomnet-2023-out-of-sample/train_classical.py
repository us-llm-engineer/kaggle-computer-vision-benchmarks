"""Methodology 1: hand-crafted image features + a closed-form multi-output ridge model.

Features per 224x128 image (no learning involved): a coarse RGB histogram (4x4x4), per-cell colour mean and std on a 2x4 grid, gradient-orientation
histograms (8 bins, magnitude weighted) per cell, the Laplacian-like energy per cell, and a 16x8 grey thumbnail. The ridge model regresses the
multi-hot label vector; its output is used as a ranking score for MAP@20, and a negative maximum score as the out-of-sample score.
The ridge strength is chosen on the validation split from a small grid; every candidate is logged.
usage: python3 train_classical.py"""
import json
import os
import time

import numpy as np

import fn_common as C

os.makedirs(C.OUT, exist_ok=True); os.makedirs(C.CKPT, exist_ok=True); t0 = time.time()
meta, sp = C.load_prep(); Y = np.load(f"{C.PREP}/train_y.npy"); X = np.load(f"{C.PREP}/train_x224.npy", mmap_mode="r"); XE = np.load(f"{C.PREP}/eval_x224.npy", mmap_mode="r")
tr, vi, vn = sp["train"], sp["val_in"], sp["val_novel"]; val = np.r_[vi, vn]; novel = np.r_[np.zeros(len(vi), bool), np.ones(len(vn), bool)]


def features(batch):
    out = []
    for start in range(0, len(batch), 256):
        x = np.asarray(batch[start:start + 256]).astype(np.float32) / 255.0; n, h, w, _ = x.shape; f = []
        q = np.minimum((x * 4).astype(int), 3); code = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        f.append(np.stack([np.bincount(c.ravel(), minlength=64) for c in code]) / (h * w))
        cells = x.reshape(n, 2, h // 2, 4, w // 4, 3); f.append(cells.mean((2, 4)).reshape(n, -1)); f.append(cells.std((2, 4)).reshape(n, -1))
        g = x.mean(-1); gy, gx = np.gradient(g, axis=(1, 2)); mag = np.hypot(gx, gy); ang = ((np.arctan2(gy, gx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        mc, ac = mag.reshape(n, 2, h // 2, 4, w // 4), ang.reshape(n, 2, h // 2, 4, w // 4)
        hist = np.stack([(mc * (ac == b)).sum((2, 4)) for b in range(8)], -1).reshape(n, -1); f.append(hist / (h * w))
        lap = np.abs(g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
        f.append(lap[:, :124, :220].reshape(n, 2, 62, 4, 55).mean((2, 4)).reshape(n, -1))
        f.append(g.reshape(n, 16, h // 16, 8, w // 8).mean((2, 4)).reshape(n, -1))
        out.append(np.concatenate(f, 1))
    return np.concatenate(out)


F_tr, F_val, F_ev = features(X[tr]), features(X[val]), features(XE)
mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6; F_tr, F_val, F_ev = [(a - mu) / sd for a in (F_tr, F_val, F_ev)]
print(f"features {F_tr.shape[1]} dims for {len(tr)} train / {len(val)} validation / {len(F_ev)} eval images ({time.time() - t0:.0f}s)", flush=True)

T = Y[tr].astype(np.float64); A = F_tr.T @ F_tr; B = F_tr.T @ (T - T.mean(0)); trials = []; best = None
for alpha in (1, 10, 100, 1000, 10000, 30000):
    W = np.linalg.solve(A + alpha * np.eye(A.shape[0]), B); logit = lambda F: F @ W + T.mean(0)
    s_val = logit(F_val); m = C.score_split(np.log(np.clip(s_val, 1e-4, 1 - 1e-4) / (1 - np.clip(s_val, 1e-4, 1 - 1e-4))), Y[val], novel)
    s_tr = logit(F_tr[:1500]); m["train_map20"] = C.map_at_k(s_tr, Y[tr][:1500]); m["alpha"] = alpha; trials.append(m)
    print(f"alpha {alpha:>6}: val MAP@20 {m['map20']:.4f}  sAUC(msp) {m['sauc_msp']:.4f}  train MAP@20 {m['train_map20']:.4f}", flush=True)
    if best is None or m["proxy_score"] > best[0]["proxy_score"]: best = (m, W.copy())
m, W = best; clipped = lambda F: np.clip(F @ W + T.mean(0), 1e-4, 1 - 1e-4)
to_logit = lambda p: np.log(p / (1 - p))
np.savez_compressed(f"{C.OUT}/classical_val.npz", logits=to_logit(clipped(F_val)).astype(np.float32), val_idx=val, novel=novel)
np.savez_compressed(f"{C.OUT}/classical_test.npz", logits=to_logit(clipped(F_ev)).astype(np.float32), ids=np.load(f"{C.PREP}/eval_ids.npy"))
np.savez(f"{C.CKPT}/classical_best.npz", W=W, mu=mu, sd=sd, bias=T.mean(0), alpha=m["alpha"])
json.dump({"name": "classical", "best": m, "trials": trials, "feature_dims": int(F_tr.shape[1]), "seconds": round(time.time() - t0, 1)}, open(f"{C.OUT}/classical_history.json", "w"), indent=1)
print("best", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), f"({time.time() - t0:.0f}s)")

"""Methodologies 2 and 3, trained from scratch (no pre-trained weights) with one script:

  cnn  : residual CNN on 224x128 images, global average + max pooling, one logit per category.
  mil  : the 448x256 image is cut into a 2x4 grid of 128x112 tiles; a small shared CNN scores each tile and the tile logits are combined by a
         log-sum-exp pooling (multiple-instance learning), so a small animal in one tile can switch its category on for the whole image.

Data live on the GPU as uint8; augmentation (flips, zoom/shift, small rotation, brightness/contrast, noise) is done on the GPU per batch.
Loss: binary cross-entropy with a clipped per-category positive weight. Selection score on validation: 0.5 * (MAP@20 + sAUC with the max-probability
score). Everything is logged each epoch to logs/<name>.log and out/<name>_history.json (rewritten atomically), and the model, optimiser and
history are checkpointed every epoch to ckpt/<name>_last.pt (resumable) and on improvement to ckpt/<name>_best.pt.
usage: python3 train_cnn.py {cnn|mil} [--epochs N] [--batch B] [--lr LR] [--budget SECONDS]"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fn_common as C
import fn_models as M

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["cnn", "mil"]); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (resume later)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
args = ap.parse_args(); name = args.name or args.model
dev = os.environ.get("FN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
for d in (C.OUT, C.CKPT, C.LOGS): os.makedirs(d, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.backends.cudnn.benchmark = True
log_file = open(f"{C.LOGS}/{name}.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_file.write(line + "\n"); log_file.flush()


# --------------------------------------------------------------------------------------------------------------------- data
meta, sp = C.load_prep(); res = "x448" if args.model == "mil" else "x224"
Yall = np.load(f"{C.PREP}/train_y.npy"); X = np.load(f"{C.PREP}/train_{res}.npy", mmap_mode="r"); XE = np.load(f"{C.PREP}/eval_{res}.npy", mmap_mode="r")
tr, vi, vn = sp["train"], sp["val_in"], sp["val_novel"]; val = np.r_[vi, vn]; novel = np.r_[np.zeros(len(vi), bool), np.ones(len(vn), bool)]
to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
Xtr, Ytr, Xva = to_gpu(X[tr]), to_gpu(Yall[tr]).float(), to_gpu(X[val]); Yva = Yall[val]
mean = torch.tensor(meta["channel_mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(meta["channel_std"], device=dev).view(1, 3, 1, 1)
pos = Yall[tr].sum(0); pos_weight = torch.tensor(np.where(pos > 0, np.clip(np.sqrt((len(tr) - pos) / np.maximum(pos, 1)), 1, 30), 1.0), device=dev).float()
sub = np.random.RandomState(1).choice(len(tr), min(1000, len(tr)), replace=False)   # fixed training subset scored in eval mode every epoch
log(f"{name}: device {dev}, train {len(tr)}, val in-dist {len(vi)}, val novel {len(vn)}, input {tuple(Xtr.shape[1:])}, batch {args.batch}")


def to_input(xb):
    return xb.permute(0, 3, 1, 2).float().div_(255.0)


def augment(x):
    B = x.shape[0]; r = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    sx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0); sy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    s = r(0.75, 1.0); tx, ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s), (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    rot = r(-1, 1) * math.radians(10); co, si = torch.cos(rot), torch.sin(rot)
    theta = torch.stack([torch.stack([s * co * sx, -s * si, tx], 1), torch.stack([s * si, s * co * sy, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), padding_mode="reflection", align_corners=False)
    m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * r(0.75, 1.25).view(B, 1, 1, 1) + m + r(-0.2, 0.2).view(B, 1, 1, 1)).clamp(0, 1)
    return x + 0.02 * torch.randn_like(x)


# -------------------------------------------------------------------------------------------------------------------- models
model = M.build(args.model).to(dev).to(memory_format=torch.channels_last)
steps_per_epoch = math.ceil(len(tr) / args.batch)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.2)
scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
n_params = sum(p.numel() for p in model.parameters()); log(f"parameters {n_params / 1e6:.2f} M")
import re
from sklearn.metrics import average_precision_score
group_names, group_params = [], []
for pname, prm in model.named_parameters():
    m_ = re.search(r"trunk\.(\d+)\.", pname); g = f"trunk.{int(m_.group(1)):02d}" if m_ else ("head" if "fc." in pname else pname.split(".")[0])
    if g not in group_names: group_names.append(g); group_params.append([])
    group_params[group_names.index(g)].append(prm)
group_names, group_params = zip(*sorted(zip(group_names, group_params), key=lambda t: t[0])); group_names = list(group_names)
REL_EDGES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0001]   # reliability bins of the predicted probability
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def predict(data, bs=128):
    model.eval(); L, Fe = [], []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]; chunk = chunk if torch.is_tensor(chunk) else torch.from_numpy(np.ascontiguousarray(chunk)).to(dev)
        x = (to_input(chunk) - mean) / std
        with torch.autocast(dev, enabled=dev == "cuda"):
            logit, feat = model(x.contiguous(memory_format=torch.channels_last))
        L.append(logit.float().cpu()); Fe.append(feat.float().cpu())
    return torch.cat(L).numpy(), torch.cat(Fe).numpy()


# ------------------------------------------------------------------------------------------------------------ resume / loop
last, best_path, hist_path = f"{C.CKPT}/{name}_last.pt", f"{C.CKPT}/{name}_best.pt", f"{C.OUT}/{name}_history.json"
history, best_score, start_epoch = [], -1.0, 0
if os.path.exists(last):
    ck = torch.load(last, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    history, best_score, start_epoch = ck["history"], ck["best_score"], ck["epoch"]; log(f"resumed at epoch {start_epoch}, best score {best_score:.4f}")


def save_history():
    tmp = hist_path + ".tmp"; json.dump({"name": name, "args": vars(args), "params": n_params, "history": history}, open(tmp, "w")); os.replace(tmp, hist_path)


started = time.time()
for epoch in range(start_epoch, args.epochs):
    w0 = [torch.cat([p_.detach().flatten() for p_ in ps]).clone() for ps in group_params]; gacc = torch.zeros(len(group_params), device=dev); gsteps = 0
    t_ep = time.time(); model.train(); perm = torch.randperm(len(tr), device=dev); loss_sum, n_seen, gnorms, correct_top1 = 0.0, 0, [], 0.0
    for i in range(0, len(tr), args.batch):
        idx = perm[i:i + args.batch]; x = augment(to_input(Xtr[idx])); x = ((x - mean) / std).contiguous(memory_format=torch.channels_last); y = Ytr[idx]
        with torch.autocast(dev, enabled=dev == "cuda"):
            logit, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        gacc += torch.stack([torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p_.grad for p_ in ps]))) for ps in group_params]).float(); gsteps += 1
        gnorms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))); scaler.step(opt); scaler.update(); sched.step()
        loss_sum += float(loss.detach()) * len(idx); n_seen += len(idx); correct_top1 += float(y[torch.arange(len(idx)), logit.argmax(1)].sum())
    train_seconds = time.time() - t_ep
    v_logit, v_feat = predict(Xva); t_logit, _ = predict(Xtr[torch.from_numpy(sub).to(dev)])
    m = C.score_split(v_logit, Yva, novel); p = 1 / (1 + np.exp(-v_logit.astype(np.float64)))
    val_loss = float(F.binary_cross_entropy_with_logits(torch.from_numpy(v_logit[~novel]), torch.from_numpy(Yva[~novel]).float(), pos_weight=pos_weight.cpu()))
    msp = p.max(1); bins = np.linspace(0, 1, 21)
    class_ap = [float(average_precision_score(Yva[~novel][:, c], p[~novel][:, c])) if Yva[~novel][:, c].sum() > 0 else None for c in range(C.NCLS)]
    pin, yin = p[~novel].ravel(), Yva[~novel].ravel(); rel_n = np.histogram(pin, REL_EDGES)[0]; rel_pos = np.histogram(pin, REL_EDGES, weights=yin)[0]; rel_conf = np.histogram(pin, REL_EDGES, weights=pin)[0]
    gmean = (gacc / max(gsteps, 1)).cpu().numpy(); w1 = [torch.cat([p_.detach().flatten() for p_ in ps]) for ps in group_params]
    upd = [float((a_ - b_).norm() / (b_.norm() + 1e-12)) for a_, b_ in zip(w1, w0)]
    rec = {"epoch": epoch + 1, "lr": sched.get_last_lr()[0], "train_loss": loss_sum / n_seen, "val_loss": val_loss, "train_top1_running": correct_top1 / n_seen,
           "train_map20_eval": C.map_at_k(1 / (1 + np.exp(-t_logit.astype(np.float64))), Yall[tr][sub]), **{f"val_{k}": v for k, v in m.items()},
           "grad_norm_mean": float(np.mean(gnorms)), "grad_norm_max": float(np.max(gnorms)), "weight_norm": float(math.sqrt(sum(float(p_.norm()) ** 2 for p_ in model.parameters()))),
           "feat_mean": float(v_feat.mean()), "feat_std": float(v_feat.std()), "feat_zero_share": float((v_feat <= 1e-6).mean()),
           "logit_mean": float(v_logit.mean()), "logit_std": float(v_logit.std()), "logit_max_mean": float(v_logit.max(1).mean()),
           "msp_mean_in": float(msp[~novel].mean()), "msp_mean_novel": float(msp[novel].mean()),
           "msp_hist_in": np.histogram(msp[~novel], bins)[0].tolist(), "msp_hist_novel": np.histogram(msp[novel], bins)[0].tolist(),
           "class_ap": class_ap, "group_names": group_names, "grad_norm_by_group": gmean.tolist(), "update_ratio_by_group": upd,
           "rel_edges": REL_EDGES, "rel_counts": rel_n.tolist(), "rel_positives": rel_pos.tolist(), "rel_mean_conf": rel_conf.tolist(),
           "images_per_second": n_seen / train_seconds, "epoch_seconds": time.time() - t_ep,
           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0, "loss_scale": float(scaler.get_scale()) if dev == "cuda" else 1.0}
    history.append(rec); save_history()
    score = m["proxy_score"]; improved = score > best_score
    if improved:
        best_score = score; torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": m, "args": vars(args)}, best_path)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history, "best_score": best_score, "epoch": epoch + 1}, last)
    log(f"epoch {epoch + 1:>2}/{args.epochs} loss {rec['train_loss']:.4f}/{val_loss:.4f}  MAP@20 {m['map20']:.4f} (train {rec['train_map20_eval']:.4f})  sAUC msp {m['sauc_msp']:.4f} "
        f"energy {m['sauc_energy']:.4f}  proxy {score:.4f}{' *' if improved else ''}  gnorm {rec['grad_norm_mean']:.2f}  {rec['images_per_second']:.0f} img/s  {rec['epoch_seconds']:.0f}s")
    if args.budget and time.time() - started > args.budget and epoch + 1 < args.epochs:
        log(f"time budget reached after epoch {epoch + 1}; rerun to resume"); raise SystemExit(0)

# ----------------------------------------------------------------------------------------------------- final predictions
ck = torch.load(best_path, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); log(f"best epoch {ck['epoch']}: {json.dumps({k: round(v, 4) for k, v in ck['metrics'].items()})}")
v_logit, v_feat = predict(Xva); e_logit, _ = predict(XE)
np.savez_compressed(f"{C.OUT}/{name}_val.npz", logits=v_logit.astype(np.float32), feats=v_feat.astype(np.float16), val_idx=val, novel=novel, best_epoch=ck["epoch"])
np.savez_compressed(f"{C.OUT}/{name}_test.npz", logits=e_logit.astype(np.float32), ids=np.load(f"{C.PREP}/eval_ids.npy"))
log(f"saved predictions for {len(val)} validation and {len(e_logit)} eval images")

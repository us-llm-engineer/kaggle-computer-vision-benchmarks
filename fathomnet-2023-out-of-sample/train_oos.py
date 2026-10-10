"""Out-of-sample round: one trainer for the variants that train a network (V0 control, V1 hypersphere outliers, V2 radius head, V3 evidential).

  bce    V0  residual CNN, BCE with clipped positive weights (the first attempt's recipe on the new split) + supercategory head
  vmf    V1  V0 + virtual outliers drawn in the low-likelihood annulus around per-category prototypes on the unit hypersphere (VMF-GOS idea)
  radius V2  V0 + phase 2 radius head on feature mixtures of images with DISJOINT label sets (BootOOD idea)
  evid   V3  Beta evidential two-branch head with the Beta-risk loss (evidential multi-label idea)

Selection uses only the in-distribution validation images and the SELECT group (never REPORT): criterion = 0.5 * (best novelty sAUC on SELECT +
all-image MAP@20 over val_in + SELECT). Per-epoch statistics go to out/<name>_history.json (rewritten atomically) and a log line per epoch to
logs/<name>.log. Predictions for every group are saved once at the end; REPORT metrics are computed only by eval_oos.py.
usage: python3 train_oos.py {bce|vmf|radius|evid} [--epochs N] [--seed S] [--name NAME] [--smoke]"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import fn_common as C
import fn_oos_common as O
import fn_oos_models as OM

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["bce", "vmf", "radius", "evid"])
ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
ap.add_argument("--w_sc", type=float, default=0.3, help="weight of the supercategory head loss")
ap.add_argument("--lam_ood", type=float, default=0.1, help="V1/V2 outlier loss weight after its linear warm-up")
ap.add_argument("--ood_start", type=int, default=10, help="epoch at which the outlier loss starts (V1: prototypes need a trunk; V2: phase 2)")
ap.add_argument("--warm", type=int, default=5, help="epochs of linear warm-up of the outlier loss")
ap.add_argument("--patience", type=int, default=18, help="early stopping: epochs without improvement of the smoothed criterion")
ap.add_argument("--min_epochs", type=int, default=30); ap.add_argument("--warm_epochs", type=int, default=3)
ap.add_argument("--plateau_patience", type=int, default=6, help="epochs without improvement before the learning rate is cut by 0.3")
ap.add_argument("--smooth", type=int, default=3, help="criterion = mean of the last N epochs (the SELECT group is small)")
ap.add_argument("--sig_lo", type=float, default=2.0); ap.add_argument("--sig_hi", type=float, default=3.0)      # V1 annulus in sigma of chi-square
ap.add_argument("--ema", type=float, default=0.9)                                                               # V1 prototype momentum
ap.add_argument("--mix_alpha", type=float, default=1.0); ap.add_argument("--gamma", type=float, default=0.2)   # V2 mixup Beta(a, a), margin fraction
ap.add_argument("--shells", type=int, default=4); ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--smoke", action="store_true", help="synthetic images, tiny subsets: code-path check only")
ap.add_argument("--budget", type=float, default=0)
args = ap.parse_args(); name = args.name or f"{args.variant}_s{args.seed}"
dev = os.environ.get("FN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
for d in (C.OUT, C.CKPT, C.LOGS): os.makedirs(d, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.backends.cudnn.benchmark = True
if dev == "cuda": torch.set_float32_matmul_precision("high")   # TF32 matmuls on the L4; on a CPU this setting selects slow emulated bf16
log_file = open(f"{C.LOGS}/{name}.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_file.write(line + "\n"); log_file.flush()


# ---------------------------------------------------------------------------------------------------------------------- data
Yall = np.load(f"{C.PREP}/train_y.npy"); z = np.load(f"{C.PREP}/split_oos.npz"); sp = {k: z[k] for k in ("train", "val_in", "report", "select", "pool")}
meta = json.load(open(f"{C.PREP}/meta_oos.json"))
sc_of, sc_names = C.supercat_index()
SC = np.zeros((C.NCLS, len(sc_names)), np.float32); SC[np.arange(C.NCLS), sc_of] = 1.0
if args.smoke:
    rng = np.random.RandomState(0); keep = {k: v[:96] for k, v in sp.items()}; sp = keep
    X = rng.randint(0, 255, size=(int(Yall.shape[0]), 128, 224, 3), dtype=np.uint8); XE = X[:64]; args.epochs = min(args.epochs, 3); args.ood_start = 1; args.warm = 1; args.min_epochs = 1; args.patience = 99; args.warm_epochs = 1
else:
    X = np.load(f"{C.PREP}/train_x224.npy", mmap_mode="r"); XE = np.load(f"{C.PREP}/eval_x224.npy", mmap_mode="r")
tr = sp["train"]; sel = np.r_[sp["val_in"], sp["select"]]; sel_novel = np.r_[np.zeros(len(sp["val_in"]), bool), np.ones(len(sp["select"]), bool)]
gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
Xtr, Ytr, Xsel = gpu(X[tr]), gpu(Yall[tr]).float(), gpu(X[sel]); Ysel = Yall[sel]
Ysc_tr = (Ytr @ torch.from_numpy(SC).to(dev)).clamp(max=1)
mean = torch.tensor(meta["channel_mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(meta["channel_std"], device=dev).view(1, 3, 1, 1)
posc = Yall[tr].sum(0); seen = np.where(posc > 0)[0]; seen_t = torch.from_numpy(seen).to(dev)
pos_weight = torch.tensor(np.where(posc > 0, np.clip(np.sqrt((len(tr) - posc) / np.maximum(posc, 1)), 1, 30), 1.0), device=dev).float()
split_stats = {"train": int(len(tr)), "val_in": int(len(sp["val_in"])), "select": int(len(sp["select"])), "pool": int(len(sp["pool"])), "report": int(len(sp["report"])),
               "seen_categories": int(len(seen)), "unseen_columns": int(C.NCLS - len(seen)), "train_labels_per_image": float(Yall[tr].sum(1).mean()),
               "train_categories_le5": int(((posc > 0) & (posc <= 5)).sum()), "train_largest_category": int(posc.max()),
               "select_mixed_share": float(O.mixed_mask(Yall, sp["select"], seen).mean()), "pool_mixed_share": float(O.mixed_mask(Yall, sp["pool"], seen).mean()),
               "report_mixed_share": float(O.mixed_mask(Yall, sp["report"], seen).mean()), "seen_supercategories": int(len(set(sc_of[seen].tolist())))}
log(f"{name}: device {dev}, train {len(tr)}, val_in {len(sp['val_in'])}, select {len(sp['select'])}, seen categories {len(seen)}, variant {args.variant}, seed {args.seed}")


def to_input(xb): return xb.permute(0, 3, 1, 2).float().div_(255.0)


def augment(x):
    B = x.shape[0]; r = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    sx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0); sy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    s = r(0.75, 1.0); tx, ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s), (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    rot = r(-1, 1) * math.radians(10); co, si = torch.cos(rot), torch.sin(rot)
    theta = torch.stack([torch.stack([s * co * sx, -s * si, tx], 1), torch.stack([s * si, s * co * sy, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), padding_mode="reflection", align_corners=False)
    m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * r(0.75, 1.25).view(B, 1, 1, 1) + m + r(-0.2, 0.2).view(B, 1, 1, 1)).clamp(0, 1)
    return x + 0.02 * torch.randn_like(x)


# ---------------------------------------------------------------------------------------------------------------------- model and state
model = OM.OOSNet("evid" if args.variant == "evid" else "bce", hidden=args.hidden, n_shells=args.shells).to(dev).to(memory_format=torch.channels_last)
D = model.dim
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps = math.ceil(len(tr) / args.batch)
lr_scale, gstep, stale, plateau, crit_hist = 1.0, 0, 0, 0, []   # linear warm-up, then plateau-driven cuts; early stopping on the smoothed criterion
scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
n_params = sum(p.numel() for p in model.parameters()); log(f"parameters {n_params / 1e6:.2f} M")
# V1 state: per-category prototype directions, concentrations, feature scale
mu = torch.zeros(C.NCLS, D, device=dev); kappa = torch.full((C.NCLS,), 50.0, device=dev); has = torch.zeros(C.NCLS, dtype=torch.bool, device=dev); fscale = torch.tensor(1.0, device=dev)
# V2 state: in-distribution centre and reference radius
centre = torch.zeros(D, device=dev); rref = torch.tensor(1.0, device=dev); centre_init = False
beta_dist = torch.distributions.Beta(torch.tensor(float(args.mix_alpha), device=dev), torch.tensor(float(args.mix_alpha), device=dev))


def predict(data, bs=128, tta=True):
    """logits, features, supercategory logits (and alpha, beta for V3), averaged over the image and its horizontal flip."""
    model.eval(); acc = {}
    with torch.no_grad():
        for i in range(0, len(data), bs):
            x = (to_input(data[i:i + bs] if torch.is_tensor(data) else gpu(data[i:i + bs])) - mean) / std
            outs = []
            for xx in ((x, x.flip(3)) if tta else (x,)):
                with torch.autocast(dev, enabled=dev == "cuda"):
                    o = model(xx.contiguous(memory_format=torch.channels_last))
                outs.append({k: v.float() for k, v in o.items()})
            for k in outs[0]: acc.setdefault(k, []).append(torch.stack([o[k] for o in outs]).mean(0).cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def lam_ood(epoch):
    if args.variant not in ("vmf", "radius") or epoch < args.ood_start: return 0.0
    return args.lam_ood * min(1.0, (epoch - args.ood_start + 1) / max(args.warm, 1))


# ---------------------------------------------------------------------------------------------------------------------- resume
last, best_path, hist_path = f"{C.CKPT}/{name}_last.pt", f"{C.CKPT}/{name}_best.pt", f"{C.OUT}/{name}_history.json"
history, best_score, start_epoch = [], -1.0, 0
if os.path.exists(last):
    ck = torch.load(last, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); lr_scale, gstep, stale, plateau, crit_hist = ck["lr_scale"], ck["gstep"], ck["stale"], ck["plateau"], ck["crit_hist"]
    history, best_score, start_epoch = ck["history"], ck["best_score"], ck["epoch"]
    mu, kappa, has, fscale, centre, rref = (ck[k].to(dev) for k in ("mu", "kappa", "has", "fscale", "centre", "rref")); centre_init = bool(ck["centre_init"])
    log(f"resumed at epoch {start_epoch}, best score {best_score:.4f}")


def save_history():
    tmp = hist_path + ".tmp"; json.dump({"name": name, "args": vars(args), "params": n_params, "split_stats": split_stats, "history": history}, open(tmp, "w"), default=float); os.replace(tmp, hist_path)


def gname(n):
    parts = n.split("."); return ".".join(parts[:2]) if parts[0] == "trunk" else parts[0]
groups = {}
for n_, p_ in model.named_parameters(): groups.setdefault(gname(n_), []).append(p_)
unseen_cols = np.setdiff1d(np.arange(C.NCLS), seen); l2n = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
sub = np.random.RandomState(1).choice(len(tr), min(1000, len(tr)), replace=False); Xsub = Xtr[torch.from_numpy(sub).to(dev)]; Ysub = Yall[tr][sub]; Xpool = gpu(X[sp["pool"]])
sel_mixed = O.mixed_mask(Yall, sp["select"], seen); pool_mixed = O.mixed_mask(Yall, sp["pool"], seen); SCn = SC


def out_layer():
    """(weight, bias) of the category output layer: the linear head, or the positive-evidence branch of V3."""
    lay = model.pos[-1] if args.variant == "evid" else model.fc
    return lay.weight.detach(), lay.bias.detach()


def gstats(P, Yg):
    """What the network does with one group of images: confidence, label count, behaviour of never-positive outputs, supercategory hit."""
    lg = P["logit"].astype(np.float64); pr = 1 / (1 + np.exp(-lg)); ysc = np.minimum(Yg.astype(np.float32) @ SCn, 1)
    d = {"mean_max_prob": float(pr.max(1).mean()), "labels_above_half": float((pr > 0.5).sum(1).mean()), "unseen_logit_mean": float(lg[:, unseen_cols].mean()),
         "unseen_logit_max_mean": float(lg[:, unseen_cols].max(1).mean()), "argmax_unseen_share": float(np.isin(lg.argmax(1), unseen_cols).mean()),
         "supercat_top1_hit": float(ysc[np.arange(len(lg)), P["sc"].argmax(1)].mean())}
    return d


def virtual_novelty(n=256):
    """V1: joint-energy novelty of freshly drawn virtual outliers, to compare with real SELECT images."""
    ready = seen_t[has[seen_t]]
    if args.variant != "vmf" or len(ready) == 0: return None
    c = ready[torch.randint(len(ready), (n,), device=dev)]; s_ = math.sqrt(2.0 * (D - 1))
    xi = torch.empty(n, device=dev).uniform_(D - 1 + args.sig_lo * s_, D - 1 + args.sig_hi * s_); t = (1 - xi / (2 * kappa[c])).clamp(-1, 1)
    v = torch.randn(n, D, device=dev); v = F.normalize(v - (v * mu[c]).sum(1, keepdim=True) * mu[c], dim=1)
    zo = t[:, None] * mu[c] + torch.sqrt((1 - t ** 2).clamp(min=0))[:, None] * v
    model.eval()
    with torch.no_grad(): lo = model.heads(zo * fscale)["logit"].float().cpu().numpy().astype(np.float64)
    return {"novelty_mean": float(O.joint_energy_novelty(lo).mean()), "t_clamped_share": float((t <= -0.999).float().mean())}
started = time.time()
for epoch in range(start_epoch, args.epochs):
    w0 = {g: torch.cat([p.detach().flatten() for p in ps]).clone() for g, ps in groups.items()}
    model.train(); perm = torch.randperm(len(tr), device=dev); t_ep = time.time(); sums = {k_: torch.zeros((), device=dev) for k_ in ("loss", "main", "sc", "ood")}; n_seen = 0; gn = {g: torch.zeros((), device=dev) for g in groups}; steps_done = 0
    geo = {"t_mean": [], "kappa_mean": [], "radius_id": [], "radius_mix": [], "l_cls": [], "l_reg": [], "l_sep": [], "alpha_mean": [], "beta_mean": []}
    lam = lam_ood(epoch); ready = seen_t[has[seen_t]]   # categories that already have a prototype (fixed for the epoch)
    for i in range(0, len(tr), args.batch):
        idx = perm[i:i + args.batch]; x = ((augment(to_input(Xtr[idx])) - mean) / std).contiguous(memory_format=torch.channels_last); y = Ytr[idx]; ysc = Ysc_tr[idx]
        with torch.autocast(dev, enabled=dev == "cuda"):
            f = model.features(x)
        f = f.float(); out = model.heads(f)
        if args.variant == "evid":
            main = OM.beta_risk_loss(out["alpha"], out["beta"], y, pos_weight); geo["alpha_mean"].append(out["alpha"].detach().mean()); geo["beta_mean"].append(out["beta"].detach().mean())
        else:
            main = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
        l_sc = F.binary_cross_entropy_with_logits(out["sc"], ysc); l_ood = torch.zeros((), device=dev); B = len(idx)
        if args.variant == "vmf":
            with torch.no_grad():   # prototype and concentration statistics from the positives of this batch (masked updates, no host sync)
                fz = F.normalize(f.detach(), dim=1); w = y / y.sum(1, keepdim=True).clamp(min=1); S = w.t() @ fz; ws = w.sum(0); act = ws > 0
                m = S / ws.clamp(min=1e-9)[:, None]; rbar = m.norm(dim=1).clamp(1e-3, 0.999); m = F.normalize(m, dim=1)
                new_mu = torch.where(has[:, None], F.normalize(args.ema * mu + (1 - args.ema) * m, dim=1), m)
                kb = rbar * (D - rbar ** 2) / (1 - rbar ** 2); new_k = torch.where(has, args.ema * kappa + (1 - args.ema) * kb, kb).clamp(5.0, 2000.0)
                mu = torch.where(act[:, None], new_mu, mu); kappa = torch.where(act, new_k, kappa); has = has | act
                fscale = 0.9 * fscale + 0.1 * f.detach().norm(dim=1).mean()
            if lam > 0:
                if len(ready) > 0:
                    c = ready[torch.randint(len(ready), (B,), device=dev)]; s = math.sqrt(2.0 * (D - 1))
                    xi = torch.empty(B, device=dev).uniform_(D - 1 + args.sig_lo * s, D - 1 + args.sig_hi * s)
                    t = (1 - xi / (2 * kappa[c])).clamp(-1, 1); v = torch.randn(B, D, device=dev); v = v - (v * mu[c]).sum(1, keepdim=True) * mu[c]; v = F.normalize(v, dim=1)
                    zo = t[:, None] * mu[c] + torch.sqrt((1 - t ** 2).clamp(min=0))[:, None] * v
                    lo = model.heads(zo * fscale)["logit"]
                    l_ood = F.binary_cross_entropy_with_logits(lo, torch.zeros_like(lo)); geo["t_mean"].append(t.detach().mean()); geo["kappa_mean"].append(kappa[c].mean())
        if args.variant == "radius":
            with torch.no_grad():
                fm = f.detach().mean(0); centre = fm if not centre_init else 0.99 * centre + 0.01 * fm; centre_init = True
                rr = (f.detach() - centre).norm(dim=1).mean(); rref = rr if epoch == 0 and steps_done == 0 else 0.99 * rref + 0.01 * rr
            geo["radius_id"].append((f.detach() - centre).norm(dim=1).mean())
            if lam > 0:                                                  # every row gets a partner; rows without a label-disjoint partner carry weight 0
                ov = ((y @ y.t()) > 0) | torch.eye(B, dtype=torch.bool, device=dev)
                rnd = torch.rand(B, B, device=dev).masked_fill(ov, -1.0); j = rnd.argmax(1); good = (rnd.max(1).values >= 0).float(); gs_ = good.sum().clamp(min=1.0)
                a = beta_dist.sample((B, 1)).to(dev); fo = a * f + (1 - a) * f[j]
                k_s = torch.randint(args.shells, (B,), device=dev); rr_ = rref.clamp(min=1e-6); rho = 0.25 * rref + ((1 - args.gamma) * rref - 0.25 * rref) * k_s.float() / max(args.shells - 1, 1)
                rad = (fo - centre).norm(dim=1)
                l_cls = (F.cross_entropy(model.radius((fo - centre) / rr_), k_s, reduction="none") * good).sum() / gs_
                l_reg = ((((rad - rho) / rr_) ** 2) * good).sum() / gs_
                cosw = F.normalize(fo, dim=1) @ F.normalize(model.fc.weight.detach(), dim=1).t(); l_sep = (cosw.abs().mean(1) * good).sum() / gs_
                l_ood = l_cls + l_reg + l_sep
                geo["radius_mix"].append(((rad * good).sum() / gs_).detach()); geo["l_cls"].append(l_cls.detach()); geo["l_reg"].append(l_reg.detach()); geo["l_sep"].append(l_sep.detach())
        loss = main + args.w_sc * l_sc + lam * l_ood
        cur_lr = args.lr * min(1.0, (gstep + 1) / max(args.warm_epochs * steps, 1)) * lr_scale
        for pg in opt.param_groups: pg["lr"] = cur_lr
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        for g, ps in groups.items():
            gs = [p.grad.norm() for p in ps if p.grad is not None]
            if gs: gn[g] += torch.linalg.vector_norm(torch.stack(gs))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); gstep += 1; steps_done += 1
        sums["loss"] += loss.detach() * B; sums["main"] += main.detach() * B; sums["sc"] += l_sc.detach() * B; sums["ood"] += l_ood.detach() * B; n_seen += B
    sums = {k_: float(v_) for k_, v_ in sums.items()}; gn = {g: float(v_) for g, v_ in gn.items()}   # one synchronisation per epoch
    train_seconds = time.time() - t_ep
    P = predict(Xsel); alpha = P.get("alpha"); beta = P.get("beta")
    cen = centre.cpu().numpy() if args.variant == "radius" and centre_init else None
    cand = O.candidate_scores(P["logit"].astype(np.float64), P["feat"], cen, alpha, beta)
    s_in = {k: v[~sel_novel] for k, v in cand.items()}; s_sel = {k: v[sel_novel] for k, v in cand.items()}
    sauc_sel = {k: O.group_sauc(s_in[k], s_sel[k]) for k in cand}
    best_key = max(sauc_sel, key=sauc_sel.get)
    p = 1 / (1 + np.exp(-P["logit"].astype(np.float64)))
    map_seen = C.map_at_k(p[~sel_novel], Ysel[~sel_novel]); map_sel_novel = C.map_at_k(p[sel_novel], Ysel[sel_novel]); map_all = C.map_at_k(p, Ysel)
    crit = map_seen   # checkpoint and early-stopping criterion: MAP@20 on the ID validation images only (a max over scores on the small SELECT group was optimistic and stopped training early)
    bins = np.linspace(np.percentile(cand[best_key], 1), np.percentile(cand[best_key], 99), 31)
    # ---- interpretation of the split and of the out-of-sample handling (SELECT, POOL and train are read; REPORT never)
    Pt = predict(Xsub); Pp = predict(Xpool); P_in = {k: v[~sel_novel] for k, v in P.items()}; P_sel = {k: v[sel_novel] for k, v in P.items()}
    cand_t = O.candidate_scores(Pt["logit"].astype(np.float64), Pt["feat"], cen, Pt.get("alpha"), Pt.get("beta")); cand_p = O.candidate_scores(Pp["logit"].astype(np.float64), Pp["feat"], cen, Pp.get("alpha"), Pp.get("beta"))
    Yin_, Ysel_ = Ysel[~sel_novel], Ysel[sel_novel]; pr_ = lambda Pg: 1 / (1 + np.exp(-Pg["logit"].astype(np.float64)))
    w_, b_ = out_layer(); seen_w = float(w_[seen_t].norm(dim=1).mean()); un_w = float(w_[torch.from_numpy(unseen_cols).to(dev)].norm(dim=1).mean())
    Ft = Pt["feat"].astype(np.float64); wt = Ysub[:, seen].astype(np.float64); okc = wt.sum(0) > 0
    protos = l2n((wt.T @ Ft)[okc] / wt.sum(0)[okc, None]); mc = lambda Fg: (l2n(Fg.astype(np.float64)) @ protos.T).max(1)
    mc_t, mc_in, mc_sel, mc_pool = mc(Pt["feat"]), mc(P_in["feat"]), mc(P_sel["feat"]), mc(Pp["feat"])
    seen_pos = Ysel_[:, seen]; mixed_recall = float((pr_(P_sel)[sel_mixed][:, seen] * seen_pos[sel_mixed]).sum() / max(seen_pos[sel_mixed].sum(), 1))
    interp = {
        "sauc_train_vs_val": O.group_sauc(cand_t[best_key], s_in[best_key]), "sauc_pool": O.group_sauc(s_in[best_key], cand_p[best_key]),
        "sauc_select_pure": O.group_sauc(s_in[best_key], s_sel[best_key][~sel_mixed]) if (~sel_mixed).any() else None,
        "sauc_select_mixed": O.group_sauc(s_in[best_key], s_sel[best_key][sel_mixed]) if sel_mixed.any() else None,
        "sauc_select_by_score": sauc_sel, "sauc_pool_by_score": {k: O.group_sauc(s_in[k], cand_p[k]) for k in cand},
        "map": {"train": C.map_at_k(pr_(Pt), Ysub), "val_in": map_seen, "select_pure": C.map_at_k(pr_(P_sel)[~sel_mixed], Ysel_[~sel_mixed]) if (~sel_mixed).any() else None,
                "select_mixed": C.map_at_k(pr_(P_sel)[sel_mixed], Ysel_[sel_mixed]) if sel_mixed.any() else None, "pool": C.map_at_k(pr_(Pp), Yall[sp["pool"]])},
        "group": {"train": gstats(Pt, Ysub), "val_in": gstats(P_in, Yin_), "select": gstats(P_sel, Ysel_), "pool": gstats(Pp, Yall[sp["pool"]])},
        "mixed_seen_label_recall": mixed_recall,
        "proto": {"maxcos_train": float(mc_t.mean()), "maxcos_val_in": float(mc_in.mean()), "maxcos_select": float(mc_sel.mean()), "maxcos_pool": float(mc_pool.mean()),
                  "sauc_maxcos_select": O.group_sauc(-mc_in, -mc_sel), "sauc_maxcos_pool": O.group_sauc(-mc_in, -mc_pool)},
        "output_layer": {"seen_weight_norm": seen_w, "unseen_weight_norm": un_w, "seen_bias_mean": float(b_[seen_t].mean()), "unseen_bias_mean": float(b_[torch.from_numpy(unseen_cols).to(dev)].mean())},
        "virtual_outliers": virtual_novelty(), "virtual_vs_real_novelty": {"val_in": float(s_in["joint_energy"].mean()), "select": float(s_sel["joint_energy"].mean()), "pool": float(cand_p["joint_energy"].mean())},
        "radius": ({"val_in": float(np.linalg.norm(P_in["feat"] - cen, axis=1).mean()), "select": float(np.linalg.norm(P_sel["feat"] - cen, axis=1).mean()),
                    "pool": float(np.linalg.norm(Pp["feat"] - cen, axis=1).mean()), "rref": float(rref)} if cen is not None else None)}
    w1 = {g: torch.cat([q.detach().flatten() for q in ps]) for g, ps in groups.items()}
    crit_hist.append(crit); crit_s = float(np.mean(crit_hist[-args.smooth:]))
    rec = {"epoch": epoch + 1, "lr": cur_lr, "lr_scale": lr_scale, "crit_smooth": crit_s, "interp": interp, "train_loss": sums["loss"] / n_seen, "main_loss": sums["main"] / n_seen, "sc_loss": sums["sc"] / n_seen,
           "ood_loss": sums["ood"] / n_seen, "lam_ood": lam, "sel_sauc": sauc_sel, "best_score_key": best_key, "sel_map_seen": map_seen, "sel_map_novel": map_sel_novel,
           "sel_map_all": map_all, "criterion": crit, "hist_bins": bins.tolist(), "hist_in": np.histogram(s_in[best_key], bins)[0].tolist(), "hist_novel": np.histogram(s_sel[best_key], bins)[0].tolist(),
           "grad_norm_by_group": {g: v / max(steps_done, 1) for g, v in gn.items()},
           "update_ratio": {g: float((w1[g] - w0[g]).norm() / (w0[g].norm() + 1e-12)) for g in groups},
           "geometry": {k: (float(torch.stack([torch.as_tensor(x_, dtype=torch.float32, device="cpu") for x_ in v]).mean()) if v else None) for k, v in geo.items()}, "feat_norm_mean": float(np.linalg.norm(P["feat"], axis=1).mean()),
           "images_per_second": n_seen / train_seconds, "epoch_seconds": time.time() - t_ep,
           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0, "loss_scale": float(scaler.get_scale()) if dev == "cuda" else 1.0}
    eligible = args.variant in ("bce", "evid") or (epoch + 1) >= args.ood_start + args.warm
    improved = eligible and crit_s > best_score + 1e-4
    if improved: best_score, stale, plateau = crit_s, 0, 0
    elif eligible: stale += 1; plateau += 1
    if plateau >= args.plateau_patience and lr_scale > 0.01: lr_scale *= 0.3; plateau = 0; log(f"plateau: learning-rate scale cut to {lr_scale:.4f}")
    rec["stale"] = stale; history.append(rec); save_history()
    if improved: torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, best_path)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "lr_scale": lr_scale, "gstep": gstep, "stale": stale, "plateau": plateau, "crit_hist": crit_hist, "history": history, "best_score": best_score, "epoch": epoch + 1, "mu": mu.cpu(), "kappa": kappa.cpu(),
                "has": has.cpu(), "fscale": fscale.cpu(), "centre": centre.cpu(), "rref": rref.cpu(), "centre_init": centre_init}, last)
    log(f"epoch {epoch + 1:>2}/{args.epochs} loss {rec['train_loss']:.4f} main {rec['main_loss']:.4f} ood {rec['ood_loss']:.4f}  MAP seen {map_seen:.4f} novel {map_sel_novel:.4f} all {map_all:.4f}  "
        f"sAUC({best_key}) {sauc_sel[best_key]:.4f}  crit {crit:.4f} smooth {crit_s:.4f}{' *' if improved else ''} stale {stale}  lr {cur_lr:.5f}  {rec['images_per_second']:.0f} img/s  {rec['epoch_seconds']:.0f}s")
    if args.budget and time.time() - started > args.budget and epoch + 1 < args.epochs: log("time budget reached; rerun to resume"); raise SystemExit(0)
    if eligible and stale >= args.patience and epoch + 1 >= args.min_epochs: log(f"early stopping at epoch {epoch + 1}: no improvement of the smoothed criterion for {stale} epochs"); break

# ---------------------------------------------------------------------------------------------------------------------- final predictions (all groups; REPORT metrics are computed elsewhere)
ck = torch.load(best_path, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); log(f"best epoch {ck['epoch']}, criterion {best_score:.4f}")
res = {"best_epoch": ck["epoch"], "centre": centre.cpu().numpy(), "mu": mu.cpu().numpy(), "kappa": kappa.cpu().numpy(), "seen": seen, "report_classes": z["report_classes"], "select_classes": z["select_classes"], "pool_classes": z["pool_classes"]}
for g, idx in (("train", tr), ("val_in", sp["val_in"]), ("select", sp["select"]), ("pool", sp["pool"]), ("report", sp["report"])):
    P = predict(gpu(X[idx])); res[f"{g}_idx"] = idx
    for k, v in P.items(): res[f"{g}_{k}"] = v.astype(np.float16 if k == "feat" else np.float32)
P = predict(gpu(XE)); [res.__setitem__(f"eval_{k}", v.astype(np.float16 if k == "feat" else np.float32)) for k, v in P.items()]
np.savez_compressed(f"{C.OUT}/{name}_pred.npz", **res)
log(f"saved predictions of {name}")

"""Loop-2 training: pre-trained models. Same data, split, monitoring subset, selection rule, tiled inference and logging as train_fil.py (which stays untouched).
usage: python3 train_fil2.py --variant m2fx|r34|r34e|r34c --seed 0 --prep prep --out runs [--time-budget-min 30] [--init-from runs/v3_s0/best.pt] [--resume]
  m2fx  Mask2Former swin-tiny continued from the round-1 checkpoint (--init-from)   r34  ResNet34-ImageNet U-Net, BCE + soft Dice, EMA
  r34e  r34 + edge-guided bottleneck attention   r34c  r34 trained on the soft consensus of the annotator copies of an image
Every variant keeps an EMA of the weights; CHOOSE evaluation, best.pt and the instance-parameter-free monitoring use the EMA weights.
Roles come from prep/meta.json: FIT trains; CHOOSE (a fixed subset, same for every variant) drives early stopping / checkpoint choice; REPORT is never read here.
Selection criterion: exponentially smoothed mean per-record PQ on the CHOOSE subset (default instance parameters, fixed before training).
Per epoch: one log line, one atomic rewrite of <run>/history.json with losses, learning rate, per-group gradient and update ratios, CHOOSE Dice/PQ/SQ/RQ,
instance counts, fragments, train-vs-CHOOSE gap, throughput and memory; last.pt every epoch, best.pt on improvement (model, optimiser, state)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF
import torchvision.transforms.v2.functional as TV

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402
import fil_infer as I  # noqa: E402
import fil_m2f as W  # noqa: E402
import fil_models as M  # noqa: E402
import fil_pre as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["m2fx", "r34", "r34e", "r34c"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prep", default="prep")
ap.add_argument("--mode", default="flat_lcn")
ap.add_argument("--out", default="runs")
ap.add_argument("--epochs", type=int, default=150, help="cap; early stopping normally ends the run earlier")
ap.add_argument("--min-epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--crop", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--base", type=int, default=32)
ap.add_argument("--alpha", type=float, default=0.3, help="v1: weight of the soft-clDice term (fixed in advance)")
ap.add_argument("--k", type=int, default=15, help="v1: soft-skeleton iterations (>= largest structure radius; measured max radius is ~14 px)")
ap.add_argument("--steps", type=int, default=0, help="steps per epoch (0 = number of FIT records // bs)")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--eval-n", type=int, default=40, help="number of CHOOSE records in the fixed monitoring subset")
ap.add_argument("--eval-every", type=int, default=2)
ap.add_argument("--patience", type=int, default=12, help="evaluations without improvement of the smoothed criterion before stopping")
ap.add_argument("--lr-patience", type=int, default=4)
ap.add_argument("--time-budget-min", type=float, default=0.0, help="hard wall-clock budget for the training loop: cosine learning-rate decay over it, the last epoch is evaluated, then stop (0 = off)")
ap.add_argument("--cosine-epochs", type=int, default=0, help="cosine learning-rate decay over this many epochs (epoch-based schedule for a fixed-length run; 0 = use the wall-clock budget or the plateau rule)")
ap.add_argument("--cosine-offset", type=int, default=0, help="epochs already trained before this cosine cycle (a warm restart over epochs offset+1 .. offset+cosine-epochs)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--init-from", default="", help="checkpoint whose model weights initialise this run (warm start of a new run, epochs counted from 0)")
ap.add_argument("--ema", type=float, default=0.995, help="decay of the weight EMA (per step)")
ap.add_argument("--enc-lr-mult", type=float, default=0.33, help="U-Net variants: encoder learning rate = lr * this (pre-trained encoder fine-tunes gently)")
ap.add_argument("--no-aug", action="store_true")
ap.add_argument("--wd", type=float, default=0.05, help="v3 (AdamW) weight decay")
ap.add_argument("--grad-clip", type=float, default=1.0, help="v3: global gradient-norm clip")
ap.add_argument("--save-last-s", type=float, default=60.0, help="seconds between writes of last.pt (best.pt is written on every improvement)")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
run = os.path.join(args.out, f"{args.variant}_s{args.seed}")
os.makedirs(run, exist_ok=True)
json.dump(vars(args) | {"variant": args.variant}, open(os.path.join(run, "config.json"), "w"), indent=1)

meta, X, LAB, fidx = I.load_prep(args.prep, args.mode)
recs = F.load_records(F.find_root(os.environ.get("FIL_ROOT")))
recs = [r for r in recs if r["rid"] in {m["rid"] for m in meta["records"]}]
assert [r["rid"] for r in recs] == [m["rid"] for m in meta["records"]], "prep records are not in the loader's order"
fit_j = [j for j, m in enumerate(meta["records"]) if m["role"] == "fit"]
choose_all = [m["rid"] for m in meta["records"] if m["role"] == "choose"]
rng0 = np.random.RandomState(0)
eval_ids = sorted(rng0.choice(choose_all, size=min(args.eval_n, len(choose_all)), replace=False).tolist())
eval_files = sorted({m["file"] for m in meta["records"] if m["rid"] in set(eval_ids)})
copies = {}                                                    # file -> record indices of all annotator copies (all copies of a file share a role, so all are FIT here)
for j_, m_ in enumerate(meta["records"]):
    copies.setdefault(m_["file"], []).append(j_)
steps = args.steps or max(1, len(fit_j) // args.bs)
print(f"[train] run={run} dev={dev} fit_records={len(fit_j)} choose_subset={len(eval_ids)} steps/epoch={steps} bs={args.bs} crop={args.crop}", flush=True)


class CropDS(torch.utils.data.Dataset):
    """random full-resolution crops; 70% are centred near a random annotated vertex, the rest uniform; flips and 90-degree rotations.
    Index i is GLOBAL across epochs (it seeds the sampling), so the loader workers can persist for the whole run. Returns the input crop, the instance label
    crop with dense ids 1..n and n."""

    def __init__(self):
        self.x = self.lab = None

    def __len__(self):
        return 1 << 30

    def __getitem__(self, i):
        if self.x is None:
            self.x = np.load(os.path.join(args.prep, f"x_{args.mode}.npy"), mmap_mode="r")
            self.lab = np.load(os.path.join(args.prep, "lab.npy"), mmap_mode="r")
        rng = np.random.default_rng((args.seed, i))
        j = fit_j[int(rng.integers(len(fit_j)))]
        fi = fidx[("train", meta["records"][j]["file"])]
        c = args.crop
        if rng.random() < 0.7 and recs[j]["polys"]:
            p = recs[j]["polys"][int(rng.integers(len(recs[j]["polys"])))]
            v = int(rng.integers(len(p) // 2))
            cx, cy = p[2 * v] + rng.integers(-c // 4, c // 4 + 1), p[2 * v + 1] + rng.integers(-c // 4, c // 4 + 1)
            x0, y0 = int(np.clip(cx - c // 2, 0, F.S - c)), int(np.clip(cy - c // 2, 0, F.S - c))
        else:
            x0, y0 = int(rng.integers(0, F.S - c + 1)), int(rng.integers(0, F.S - c + 1))
        xc = np.asarray(self.x[fi, y0:y0 + c, x0:x0 + c])
        raw = np.asarray(self.lab[j, y0:y0 + c, x0:x0 + c])
        sib = np.zeros((3, c, c), np.uint8)                   # annotator copies of this image (own included), for the soft consensus target
        ns = 1
        if args.variant == "r34c":
            cj = copies[meta["records"][j]["file"]][:3]
            ns = len(cj)
            for q, jj in enumerate(cj):
                sib[q] = np.asarray(self.lab[jj, y0:y0 + c, x0:x0 + c]) > 0
        return torch.from_numpy(np.ascontiguousarray(xc)), torch.from_numpy(np.ascontiguousarray(raw)), torch.from_numpy(sib), ns


def gpu_batch(xq, raw, sib, nsib, gen):
    """uint8 crops (B,c,c), instance labels (B,c,c), sibling foregrounds (B,3,c,c) already on the device -> float input (B,1,c,c), dense instance labels,
    instance counts, training target (B,1,c,c). The augmentation (flips, 90-degree rotations) and the dequantisation run on the GPU with torch / torchvision ops,
    so the loader workers only slice the memory-mapped arrays."""
    B = xq.shape[0]
    if not args.no_aug:
        r = torch.rand(B, 3, generator=gen, device=xq.device)
        xs, ls, ss = [], [], []
        for b in range(B):
            x_, l_, s_ = xq[b], raw[b], sib[b]
            if r[b, 0] < 0.5:
                x_, l_, s_ = TV.horizontal_flip(x_), TV.horizontal_flip(l_), TV.horizontal_flip(s_)
            if r[b, 1] < 0.5:
                x_, l_, s_ = TV.vertical_flip(x_), TV.vertical_flip(l_), TV.vertical_flip(s_)
            k = int(r[b, 2] * 4) % 4
            xs.append(torch.rot90(x_, k, (-2, -1))); ls.append(torch.rot90(l_, k, (-2, -1))); ss.append(torch.rot90(s_, k, (-2, -1)))
        xq, raw, sib = torch.stack(xs), torch.stack(ls), torch.stack(ss)
    x = (xq.float() / 31.875 - 4.0)[:, None]                  # inverse of the quantisation, as fil_common.to_float
    if args.variant == "m2fx":                                # dense ids 1..n per crop (the mask-classification loss needs them)
        lab = torch.zeros_like(raw, dtype=torch.long); n = []
        for b in range(B):
            u, inv = torch.unique(raw[b], return_inverse=True)
            has0 = bool(u.numel() and u[0] == 0)
            lab[b] = inv if has0 else inv + 1
            n.append(u.numel() - (1 if has0 else 0))
        lab = lab.clamp_max(255).to(torch.uint8)
        return x, lab, n, (lab > 0).float()[:, None]
    if args.variant == "r34c":
        tgt = sib.float().sum(1) / nsib.float()[:, None, None]             # mean foreground of the copies that exist (an empty crop of a copy still counts)
        return x, raw, None, tgt[:, None]
    return x, raw, None, (raw > 0).float()[:, None]


class EpochSampler(torch.utils.data.Sampler):
    """yields the global sample indices of one epoch; the epoch counter lives in the main process, workers persist across epochs."""

    def __init__(self, n):
        self.n, self.epoch = n, 0

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))


model = P.build_model2({"variant": args.variant}, pretrained=not args.init_from).to(dev)
if args.init_from:                                              # warm start from a round-1 checkpoint (the file is read once; its optimiser state is not used)
    sd = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
    print("[train] init from", args.init_from, model.load_state_dict(sd), flush=True)
if args.variant != "m2fx":
    model = model.to(memory_format=torch.channels_last)
    groups = P.param_groups_r34(model)
    enc = {id(p) for p in groups["enc"]}
    opt = torch.optim.Adam([{"params": groups["enc"], "lr": args.lr * args.enc_lr_mult, "base_lr": args.lr * args.enc_lr_mult},
                            {"params": [p for p in model.parameters() if id(p) not in enc], "lr": args.lr, "base_lr": args.lr}])
else:
    groups = W.param_groups_m2f(model)
    bb = {id(p) for p in groups["encoder"]}
    opt = torch.optim.AdamW([{"params": groups["encoder"], "lr": args.lr * 0.1, "base_lr": args.lr * 0.1},
                             {"params": [p for p in model.parameters() if id(p) not in bb], "lr": args.lr, "base_lr": args.lr}], weight_decay=args.wd)
wema = P.EMA(model, args.ema)
nparams = sum(p.numel() for p in model.parameters())
state = {"epoch": 0, "hist": [], "best": -1.0, "ema": None, "bad_evals": 0, "lr_bad": 0, "best_epoch": 0}
lastp, bestp = os.path.join(run, "last.pt"), os.path.join(run, "best.pt")
if args.resume and (os.path.exists(lastp) or os.path.exists(bestp)):
    cks = [torch.load(pth, map_location=dev, weights_only=False) for pth in (lastp, bestp) if os.path.exists(pth)]
    ck = max(cks, key=lambda c: c["state"]["epoch"] if "state" in c else c["epoch"])          # the most advanced of last.pt and best.pt
    model.load_state_dict(ck["raw"]); wema.model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
    state["epoch"] = ck["epoch"]
    hp = os.path.join(run, "history.json")
    if not state["hist"] and os.path.exists(hp):                    # best.pt carries no history: take it from the history file (epochs up to the checkpoint)
        state["hist"] = [e for e in json.load(open(hp))["epochs"] if e["epoch"] <= state["epoch"]]
    if args.variant != "m2fx":                                     # a restart cycle uses the learning rates given now, not the ones saved in the optimiser state
        opt.param_groups[0]["base_lr"] = args.lr * args.enc_lr_mult; opt.param_groups[1]["base_lr"] = args.lr
    print(f"[train] resumed at epoch {state['epoch']} with {len(state['hist'])} history rows", flush=True)
print(f"[train] params={nparams:,} variant={args.variant}", flush=True)


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_hist():
    tmp = os.path.join(run, "history.json.tmp")
    json.dump({"config": vars(args), "params": nparams, "epochs": state["hist"]}, open(tmp, "w"), default=float)
    os.replace(tmp, os.path.join(run, "history.json"))


EX_IDS = eval_ids[:4]


def evaluate(want_examples=False, extra_thr=True):
    """monitoring pass on the fixed CHOOSE subset: instance metrics at the default parameters, PQ at three thresholds, recall by instance size,
    probability separation (inside vs outside the ground truth), per-record PQ, and (when asked) example crops for the qualitative figure."""
    wema.model.eval()
    ta = I.tile_args({"crop": args.crop})
    probs = {f: I.predict_prob(wema.model, X[fidx[("train", f)]], dev, **ta) for f in eval_files}
    rows = I.score_records(probs, meta, LAB, eval_ids, 0.5, 30, 0)
    s = I.summarize(rows)
    s["pq_by_thr"] = {"0.5": s["pq_mean_over_records"]}
    for th in ((0.3, 0.7) if extra_thr else ()):
        s["pq_by_thr"][str(th)] = I.summarize(I.score_records(probs, meta, LAB, eval_ids, th, 30, 0))["pq_mean_over_records"]
    ridx = {m["rid"]: j for j, m in enumerate(meta["records"])}
    pin, pout, p99 = [], [], []
    for rid in eval_ids:
        j = ridx[rid]; p = probs[meta["records"][j]["file"]].astype(np.float32); g = np.asarray(LAB[j]) > 0
        pin.append(float(p[g].mean()) if g.any() else float("nan")); pout.append(float(p[~g].mean())); p99.append(float(np.percentile(p[~g][::40], 99)))
    s["p_in_gt"], s["p_out_gt"], s["p99_out_gt"] = float(np.nanmean(pin)), float(np.mean(pout)), float(np.mean(p99))
    s["per_record_pq"] = [r["pq"] for r in rows]
    ex = None
    if want_examples:
        ex = {"rid": [], "x": [], "gt": [], "prob": [], "pred": []}
        for rid in EX_IDS:
            j = ridx[rid]; f = meta["records"][j]["file"]
            g = np.asarray(LAB[j])
            if g.max() > 0:                                   # centre on the largest annotated instance so the truth is always inside the crop
                big = 1 + int(np.argmax(np.bincount(g.ravel())[1:])); ys, xs = np.nonzero(g == big); cy, cx = int(ys.mean()), int(xs.mean())
            else:
                cy, cx = F.S // 2, F.S // 2
            y0, x0 = int(np.clip(cy - 256, 0, F.S - 512)), int(np.clip(cx - 256, 0, F.S - 512))
            sl = (slice(y0, y0 + 512), slice(x0, x0 + 512))
            pred = F.instances_from_prob(probs[f].astype(np.float32), 0.5, 30, 0)
            ex["rid"].append(rid); ex["x"].append(np.asarray(X[fidx[("train", f)]])[sl]); ex["gt"].append(g[sl]); ex["prob"].append(probs[f][sl])
            ex["pred"].append(np.minimum(pred[sl], 255).astype(np.uint8))
    model.train()
    return s, ex


def save_examples(ex, per_record):
    tmp = os.path.join(run, "examples_tmp.npz")
    np.savez_compressed(tmp, rid=np.array(ex["rid"]), x=np.stack(ex["x"]), gt=np.stack(ex["gt"]), prob=np.stack(ex["prob"]), pred=np.stack(ex["pred"]))
    os.replace(tmp, os.path.join(run, "examples.npz"))
    json.dump({"eval_ids": eval_ids, "per_record_pq": per_record}, open(os.path.join(run, "best_extra.json"), "w"))


model.train()
t_start = time.time()
budget_s = args.time_budget_min * 60.0
last_epoch_s = 0.0
last_saved = 0.0
stop_after = False
sampler = EpochSampler(steps * args.bs)
dl = torch.utils.data.DataLoader(CropDS(), batch_size=args.bs, sampler=sampler, num_workers=args.workers, drop_last=True, pin_memory=dev.type == "cuda",
                                 persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
cuda = dev.type == "cuda"
gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
for ep in range(state["epoch"] + 1, args.epochs + 1):
    t0 = time.time()
    if budget_s and (t0 - t_start) + 1.15 * last_epoch_s >= budget_s:
        stop_after = True                       # this is the last epoch that fits in the budget: it will be evaluated
    sampler.epoch = ep - 1
    agg, gstat, ustat, n_img = {}, {k: [] for k in groups}, {k: [] for k in groups}, 0      # device tensors: read ONCE per epoch (no host sync inside the step)
    if args.cosine_epochs or budget_s:          # cosine decay to 5% of each group's base learning rate: over a fixed number of epochs, else over the wall-clock budget
        frac = min(1.0, max(0.0, (ep - 1 - args.cosine_offset) / args.cosine_epochs)) if args.cosine_epochs else min(1.0, (t0 - t_start) / budget_s)
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
    lr = opt.param_groups[-1]["lr"]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for it, (xq_, rq_, sq_, ns_) in enumerate(dl):
        xb, lb, nb_, yb = gpu_batch(xq_.to(dev, non_blocking=True), rq_.to(dev, non_blocking=True), sq_.to(dev, non_blocking=True), ns_.to(dev), gen)
        if args.variant != "m2fx":
            xb = xb.contiguous(memory_format=torch.channels_last)
        # yb: hard foreground of the own copy (m2fx, r34, r34e) or the soft consensus of the copies (r34c)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cuda):
            if args.variant == "m2fx":
                loss, sem = model.train_step_outputs(xb, lb, nb_)
                parts = {}
            else:
                logits = model(xb)
        if args.variant != "m2fx":
            loss, parts = M.seg_loss(logits, yb, "v0")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.variant == "m2fx" and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if it % 10 == 0:                           # gradient / update statistics by layer group, kept as device tensors until the epoch ends
            for k, ps in groups.items():
                g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None))
                w = torch.sqrt(sum((p.detach().float() ** 2).sum() for p in ps))
                gstat[k].append(g.detach()); ustat[k].append((lr * g / (w + 1e-12)).detach())
        opt.step()
        wema.update(model, state["epoch"] * steps + it)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0) + v
        agg["loss"] = agg.get("loss", 0) + loss.detach()
        with torch.no_grad():
            pb = ((sem[:, 0] > 0.5) if args.variant == "m2fx" else (logits[:, 0].detach().float() > 0)).float()
            yh = (yb[:, 0] > 0.5).float()
            agg["train_dice"] = agg.get("train_dice", 0) + 2 * (pb * yh).sum() / (pb.sum() + yh.sum() + 1e-6)
        n_img += xb.shape[0]
    nb = it + 1
    agg = {k: float(v) / nb for k, v in agg.items()}                                        # the one synchronisation of the epoch
    gstat = {k: float(torch.stack(v).mean()) for k, v in gstat.items() if v}
    ustat = {k: float(torch.stack(v).mean()) for k, v in ustat.items() if v}
    rec = {"epoch": ep, "lr": lr, "seconds": time.time() - t0, "images_per_s": n_img / (time.time() - t0),
           "peak_mem_gib": (torch.cuda.max_memory_allocated() / 2 ** 30) if cuda else 0.0,
           **agg, "grad_norm": gstat, "update_ratio": ustat}
    if ep % args.eval_every == 0 or ep == args.epochs or stop_after:
        state["n_eval"] = state.get("n_eval", 0) + 1
        s, ex = evaluate(want_examples=True, extra_thr=(state["n_eval"] % 3 == 1 or stop_after))
        rec["choose"] = {k: s[k] for k in ("pq_mean_over_records", "sq_mean", "rq_mean", "dice_mean", "pred_instances", "gt_instances", "tp", "fp", "fn", "gt_split_total", "pred_merge_total",
                                           "pq_by_thr", "recall_by_size", "p_in_gt", "p_out_gt", "p99_out_gt")}
        rec["has_extra_thr"] = len(s["pq_by_thr"]) > 1
        rec["train_minus_choose_dice"] = rec["train_dice"] - s["dice_mean"]
        crit = s["pq_mean_over_records"]
        state["ema"] = crit if state["ema"] is None else 0.5 * crit + 0.5 * state["ema"]
        if state["ema"] > state["best"] + 1e-4:
            state["best"], state["best_epoch"], state["bad_evals"], state["lr_bad"] = state["ema"], ep, 0, 0
            save_atomic({"model": wema.model.state_dict(), "raw": model.state_dict(), "opt": opt.state_dict(), "state": state | {"hist": []}, "epoch": ep}, bestp)
            save_examples(ex, s["per_record_pq"])
        else:
            state["bad_evals"] += 1; state["lr_bad"] += 1
            if not (budget_s or args.cosine_epochs) and state["lr_bad"] >= args.lr_patience and lr > 1e-6:
                for gp in opt.param_groups:
                    gp["lr"] = gp["lr"] * 0.3
                state["lr_bad"] = 0
        rec["smoothed_pq"] = state["ema"]
    state["hist"].append(rec); state["epoch"] = ep
    if time.time() - last_saved >= args.save_last_s or stop_after:        # last.pt by time (every epoch would rewrite ~0.6 GB for the Mask2Former variant)
        save_atomic({"model": wema.model.state_dict(), "raw": model.state_dict(), "opt": opt.state_dict(), "state": state, "epoch": ep}, lastp); last_saved = time.time()
    write_hist()
    ch = rec.get("choose")
    print(f"[train] {args.variant} s{args.seed} ep {ep:3d} loss {rec['loss']:.4f} train_dice {rec['train_dice']:.3f} lr {lr:.1e} "
          + (f"| CHOOSE pq {ch['pq_mean_over_records']:.4f} dice {ch['dice_mean']:.3f} pred/gt {ch['pred_instances']}/{ch['gt_instances']} smooth {rec['smoothed_pq']:.4f} best {state['best']:.4f}@{state['best_epoch']} " if ch else "")
          + f"| {rec['seconds']:.0f}s {rec['images_per_s']:.1f} img/s mem {rec['peak_mem_gib']:.1f}G", flush=True)
    last_epoch_s = time.time() - t0
    if stop_after:
        print(f"[train] time budget of {args.time_budget_min:.1f} min reached after epoch {ep}", flush=True)
        break
    if ep >= args.min_epochs and state["bad_evals"] >= args.patience:
        print(f"[train] early stop at epoch {ep}: no improvement of the smoothed CHOOSE PQ for {state['bad_evals']} evaluations", flush=True)
        break
print(f"[train] done in {(time.time() - t_start) / 60:.1f} min; best smoothed CHOOSE PQ {state['best']:.4f} at epoch {state['best_epoch']}", flush=True)
open(os.path.join(run, "DONE"), "w").write(f"epochs={state['epoch']} best_epoch={state['best_epoch']} best_smoothed_pq={state['best']:.6f}\n")

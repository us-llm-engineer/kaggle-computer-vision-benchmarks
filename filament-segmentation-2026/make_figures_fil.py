"""Draw exactly six figures from the saved statistics (runs/*/history.json, eval.json, examples.npz). Nothing is drawn inside the training loop.
usage: python3 helpers/make_figures_fil.py --runs runs --out figures [--human-pq 0.359]
  fig1_training_dynamics.png    losses, train vs CHOOSE Dice, generalisation gap per variant (seed mean +- range)
  fig2_pq_and_final.png         CHOOSE PQ per epoch (raw and smoothed, best epoch marked) and final PQ / SQ / RQ on CHOOSE and REPORT with the human-vs-human reference
  fig3_instance_diagnostics.png predicted / true instance count, false positives and negatives, splits and merges per epoch
  fig4_optimisation_health.png  gradient norm and update ratio by layer group, learning rate
  fig5_failure_anatomy.png      recall by instance size, PQ versus threshold, probability separation inside / outside the truth
  fig6_qualitative.png          input, truth, probability and matched / unmatched instances on fixed CHOOSE crops, per variant"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="figures")
ap.add_argument("--human-pq", type=float, default=0.359, help="median PQ between annotator copies of one image (data investigation)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

COL = {"v0": "#4c78a8", "v1": "#e45756", "v2": "#54a24b", "v3": "#b279a2"}
NAME = {"v0": "V0 U-Net (control)", "v1": "V1 + soft-clDice", "v2": "V2 + edge-guided attention", "v3": "V3 Mask2Former (pre-trained)"}
runs = {}
for d in sorted(glob.glob(os.path.join(args.runs, "v*_s*"))):
    v = os.path.basename(d).split("_")[0]
    hp = os.path.join(d, "history.json")
    if os.path.exists(hp):
        runs.setdefault(v, []).append({"dir": d, "hist": json.load(open(hp))["epochs"],
                                       "eval": json.load(open(os.path.join(d, "eval.json"))) if os.path.exists(os.path.join(d, "eval.json")) else None})
variants = [v for v in ("v0", "v1", "v2", "v3") if v in runs]
assert variants, f"no runs under {args.runs}"


def series(v, getter, only_eval=False):
    """list over seeds of (epochs, values); getter(rec) -> value or None."""
    out = []
    for r in runs[v]:
        xs, ys = [], []
        for h in r["hist"]:
            y = getter(h)
            if y is not None:
                xs.append(h["epoch"]); ys.append(y)
        out.append((np.array(xs), np.array(ys, dtype=float)))
    return out


def band(ax, v, getter, label=None, **kw):
    ss = series(v, getter)
    if not any(len(x) for x, _ in ss):
        return
    n = min(len(x) for x, _ in ss if len(x)) if any(len(x) for x, _ in ss) else 0
    xs = ss[0][0][:n]
    Y = np.stack([y[:n] for _, y in ss if len(y) >= n])
    ax.plot(xs, Y.mean(0), color=COL[v], label=label or NAME[v], **kw)
    if len(Y) > 1:
        ax.fill_between(xs, Y.min(0), Y.max(0), color=COL[v], alpha=0.18)


ch = lambda k: (lambda h: h["choose"][k] if "choose" in h else None)

# 1 -------------------------------------------------------------------------------------------------------- training dynamics
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for v in variants:
    band(ax[0], v, lambda h: h["loss"])
    band(ax[1], v, lambda h: h["train_dice"], ls="-")
    band(ax[1], v, ch("dice_mean"), label=NAME[v] + " (CHOOSE)", ls="--")
    band(ax[2], v, lambda h: h.get("train_minus_choose_dice"))
ax[0].set_title("training loss (total)"); ax[1].set_title("Dice: training crops (solid) vs CHOOSE images (dashed)"); ax[2].set_title("generalisation gap: train Dice - CHOOSE Dice")
ax[2].axhline(0, color="k", lw=0.5)
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("Fig 1. Training dynamics (mean over seeds, band = min-max)"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig1_training_dynamics.png"), dpi=130); plt.close(fig)

# 2 -------------------------------------------------------------------------------------------------------- PQ and final
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
for v in variants:
    band(ax[0], v, ch("pq_mean_over_records"), marker="o", ms=3)
    for r in runs[v]:
        ev = [(h["epoch"], h["smoothed_pq"]) for h in r["hist"] if "smoothed_pq" in h]
        if ev:
            ax[0].plot(*zip(*ev), color=COL[v], ls=":", lw=1)
            be = max(ev, key=lambda t: t[1]); ax[0].scatter([be[0]], [be[1]], color=COL[v], marker="*", s=90, zorder=5)
ax[0].axhline(args.human_pq, color="gray", ls="--", lw=1); ax[0].text(0.02, args.human_pq + 0.005, "median PQ between two annotator copies", fontsize=7, color="gray", transform=ax[0].get_yaxis_transform())
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("PQ (CHOOSE subset)"); ax[0].set_title("CHOOSE PQ per epoch (solid), smoothed criterion (dotted), best (*)", fontsize=10); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
w, x0 = 0.8 / max(1, sum(len(runs[v]) for v in variants)), np.arange(3)
labs = ["PQ", "SQ", "RQ"]
k = 0
for v in variants:
    for r in runs[v]:                                   # best checkpoint of each run on the CHOOSE monitoring subset (REPORT, when present, is drawn in the next loop)
        ev = [h for h in r["hist"] if "smoothed_pq" in h]
        if not ev:
            continue
        c = max(ev, key=lambda h: h["smoothed_pq"])["choose"]
        ax[1].bar(x0 + k * w, [c["pq_mean_over_records"], c["sq_mean"], c["rq_mean"]], w, color=COL[v], alpha=0.85,
                  label=f"{NAME[v]} {os.path.basename(r['dir'])}" if True else None)
        k += 1
ax[1].axhline(args.human_pq, color="gray", ls="--", lw=1)
ax[1].set_xticks(x0 + w * (k - 1) / 2); ax[1].set_xticklabels(labs); ax[1].set_title("best checkpoint of each run (PQ = SQ x RQ)", fontsize=10); ax[1].legend(fontsize=5, ncol=2); ax[1].grid(alpha=0.3, axis="y")
fig.suptitle("Fig 2. Panoptic quality over training and at the end"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig2_pq_and_final.png"), dpi=130); plt.close(fig)

# 3 -------------------------------------------------------------------------------------------------------- instance diagnostics
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for v in variants:
    band(ax[0], v, lambda h: h["choose"]["pred_instances"] / max(h["choose"]["gt_instances"], 1) if "choose" in h else None)
    band(ax[1], v, lambda h: h["choose"]["fp"] if "choose" in h else None, ls="-")
    band(ax[1], v, lambda h: h["choose"]["fn"] if "choose" in h else None, ls="--", label=NAME[v] + " FN")
    band(ax[2], v, ch("gt_split_total"), ls="-")
    band(ax[2], v, ch("pred_merge_total"), ls="--", label=NAME[v] + " merges")
ax[0].axhline(1, color="k", lw=0.6); ax[0].set_title("predicted / true instance count"); ax[1].set_title("false positives (solid) and false negatives (dashed)")
ax[2].set_title("true instances split (solid) / predictions merging two (dashed)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("Fig 3. Instance-level diagnostics on the CHOOSE subset"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig3_instance_diagnostics.png"), dpi=130); plt.close(fig)

# 4 -------------------------------------------------------------------------------------------------------- optimisation health
fig, ax = plt.subplots(1, len(variants) + 1, figsize=(4.6 * (len(variants) + 1), 4))
for i, v in enumerate(variants):
    groups = sorted({g for r in runs[v] for h in r["hist"] for g in h.get("grad_norm", {})})
    for g in groups:
        xs = [h["epoch"] for h in runs[v][0]["hist"] if g in h.get("grad_norm", {})]
        ax[i].plot(xs, [h["grad_norm"][g] for h in runs[v][0]["hist"] if g in h.get("grad_norm", {})], label=f"{g} grad")
        ax[i].plot(xs, [h["update_ratio"][g] for h in runs[v][0]["hist"] if g in h.get("update_ratio", {})], ls="--", label=f"{g} update/weight")
    ax[i].set_yscale("log"); ax[i].set_title(NAME[v] + " (first seed)"); ax[i].set_xlabel("epoch"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=6, ncol=2)
for v in variants:
    band(ax[-1], v, lambda h: h["lr"])
ax[-1].set_yscale("log"); ax[-1].set_title("learning rate"); ax[-1].set_xlabel("epoch"); ax[-1].grid(alpha=0.3)
fig.suptitle("Fig 4. Optimisation health: gradient norm and update/weight ratio by layer group"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig4_optimisation_health.png"), dpi=130); plt.close(fig)

# 5 -------------------------------------------------------------------------------------------------------- failure anatomy
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
bins = ["300-1000", "1000-3000", ">=3000"]   # the bin below 300 px holds no ground-truth instance of the monitoring subsets
for i, v in enumerate(variants):
    last = [next((h["choose"] for h in reversed(r["hist"]) if "choose" in h and h.get("has_extra_thr", True)), None) for r in runs[v]]
    last = [l for l in last if l]
    if not last:
        continue
    rec = np.array([[b["recall"] if b["recall"] is not None else np.nan for b in l["recall_by_size"]][1:] for l in last], dtype=float)
    ax[0].bar(np.arange(3) + (i - (len(variants) - 1) / 2) * 0.2, np.nanmean(rec, 0), 0.2, color=COL[v], label=NAME[v])
    ax[1].plot([0.3, 0.5, 0.7], [np.mean([l["pq_by_thr"][k] for l in last]) for k in ("0.3", "0.5", "0.7")], marker="o", color=COL[v], label=NAME[v])
    for key, ls in (("p_in_gt", "-"), ("p_out_gt", "--"), ("p99_out_gt", ":")):
        band(ax[2], v, ch(key), ls=ls, label=NAME[v] + " " + key)
ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels([b + " px" for b in bins]); ax[0].set_ylim(0, 1); ax[0].set_title("recall (IoU > 0.5) by instance area", fontsize=10); ax[0].legend(fontsize=7, loc="lower right")
ax[1].set_xlabel("probability threshold"); ax[1].set_title("PQ versus threshold", fontsize=10); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
ax[2].set_xlabel("epoch"); ax[2].set_title("probability inside truth (-), outside (--), 99th pct outside (:)", fontsize=9); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=5, ncol=2, loc="center right")
fig.suptitle("Fig 5. Where the model fails: instance size, threshold sensitivity, score separation"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig5_failure_anatomy.png"), dpi=130); plt.close(fig)

# 6 -------------------------------------------------------------------------------------------------------- qualitative
rows_ex = []
for v in variants:
    p = os.path.join(runs[v][0]["dir"], "examples.npz")
    if os.path.exists(p):
        z = np.load(p)
        for e in range(min(2, len(z["rid"]))):
            rows_ex.append((v, str(z["rid"][e]), z["x"][e], z["gt"][e], z["prob"][e].astype(np.float32), z["pred"][e]))
fig, ax = plt.subplots(max(1, len(rows_ex)), 4, figsize=(13, 3.1 * max(1, len(rows_ex))), squeeze=False)
for r, (v, rid, x, gt, prob, pred) in enumerate(rows_ex):
    res = F.pq(gt.astype(np.int32), pred.astype(np.int32))
    ov = np.stack([x / 255.0] * 3, -1)
    for j in range(1, int(pred.max()) + 1):
        ov[pred == j] = ov[pred == j] * 0.4 + np.array([0.0, 0.8, 0.0] if res["pred_hit"][j - 1] else [0.9, 0.1, 0.1]) * 0.6
    for i in range(1, int(gt.max()) + 1):
        if res["gt_iou"][i - 1] == 0:
            m = gt == i
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            ov[edge] = [0.2, 0.4, 1.0]
    gtv = np.stack([x / 255.0] * 3, -1); gtv[gt > 0] = gtv[gt > 0] * 0.5 + np.array([1.0, 0.8, 0.0]) * 0.5
    for a, im, t in zip(ax[r], (x, gtv, prob, ov), ("input (preprocessed)", "ground truth", "probability", f"{NAME[v]}: TP green, FP red, missed GT blue (PQ {res['pq']:.2f})")):
        a.imshow(im, cmap="gray" if im.ndim == 2 and im is x else ("viridis" if im.ndim == 2 else None), vmin=None); a.set_title(t, fontsize=7); a.axis("off")
    ax[r][0].set_ylabel(rid, fontsize=6)
fig.suptitle("Fig 6. Fixed CHOOSE crops (512 x 512 around the annotated filaments)"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig6_qualitative.png"), dpi=130); plt.close(fig)
print("wrote", sorted(f for f in os.listdir(args.out) if f.startswith("fig")))

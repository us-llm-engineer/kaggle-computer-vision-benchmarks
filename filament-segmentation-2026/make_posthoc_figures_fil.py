"""Five post-training figures that interpret the final metrics, drawn from the saved statistics only (history.json, best_extra.json, data investigation).
usage: python3 helpers/make_posthoc_figures_fil.py --runs results/runs --human results/data_investigation.json --out figures
  post1_final_pq_and_ceiling.png     best smoothed CHOOSE PQ of every run against the agreement between two human annotators
  post2_pq_decomposition.png         PQ = SQ x RQ at the best checkpoint: which factor limits the score
  post3_per_record_pq.png            per-image PQ at the best checkpoint on the records all runs share, with the human-vs-human quartiles
  post4_instance_error_anatomy.png   matched, extra and missed instances, splits and merges per true instance
  post5_recall_by_size.png           recall of true instances by area, with the number of instances per bin
A CHOOSE monitoring subset is 12 to 16 images; the 12 shared by all runs are used in post3. These are not held-out REPORT scores."""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--human", default="results/data_investigation.json")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
COL = {"v0": "#4c78a8", "v1": "#e45756", "v2": "#54a24b", "v3": "#b279a2"}
NAME = {"v0": "V0 U-Net", "v1": "V1 + soft-clDice", "v2": "V2 + edge attention", "v3": "V3 Mask2Former (pre-trained)"}
H = json.load(open(args.human))["annotator_agreement"]
hq = H["pq_between_copies"]; hsq = H["sq"]; hrq = H["rq"]
R = {}
for d in sorted(glob.glob(os.path.join(args.runs, "v*_s*"))):
    name = os.path.basename(d); v = name.split("_")[0]
    ep = json.load(open(os.path.join(d, "history.json")))["epochs"]
    ev = [e for e in ep if "smoothed_pq" in e]; b = max(ev, key=lambda e: e["smoothed_pq"])
    ex = json.load(open(os.path.join(d, "best_extra.json")))
    R[name] = {"v": v, "epochs": ep[-1]["epoch"], "best_epoch": b["epoch"], "smoothed": b["smoothed_pq"], "c": b["choose"], "ids": ex["eval_ids"], "pq": ex["per_record_pq"]}
names = list(R)
vs = [v for v in ("v0", "v1", "v2", "v3") if any(R[n]["v"] == v for n in names)]

# 1 ---------------------------------------------------------------------------------------------- final PQ and the human ceiling
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, v in enumerate(vs):
    xs = [R[n]["smoothed"] for n in names if R[n]["v"] == v]
    ax.scatter([i] * len(xs), xs, s=70, color=COL[v], zorder=3)
    ax.plot([i - 0.22, i + 0.22], [np.mean(xs)] * 2, color=COL[v], lw=2)
    ax.text(i, min(xs) - 0.012, f"n={len(xs)}\nmean {np.mean(xs):.3f}", ha="center", va="top", fontsize=8)
for q, ls, lab in (("50", "--", "median PQ between two annotator copies"), ("90", ":", "90th percentile")):
    ax.axhline(hq[q], color="gray", ls=ls, lw=1.2); ax.text(len(vs) - 0.45, hq[q] + 0.004, f"{lab} ({hq[q]:.2f})", ha="right", fontsize=8, color="gray")
ax.set_xticks(range(len(vs))); ax.set_xticklabels([NAME[v] for v in vs], fontsize=9); ax.set_ylim(0.2, 0.62); ax.set_ylabel("best smoothed PQ on the CHOOSE monitoring subset"); ax.grid(alpha=0.3, axis="y")
ax.set_title("Post 1. Final score of every run against the human-vs-human ceiling")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post1_final_pq_and_ceiling.png"), dpi=130); plt.close(fig)

# 2 ---------------------------------------------------------------------------------------------- PQ = SQ x RQ
fig, ax = plt.subplots(figsize=(10, 4.6))
w = 0.8 / len(names)
for k, n in enumerate(names):
    c = R[n]["c"]
    ax.bar(np.arange(3) + k * w, [c["sq_mean"], c["rq_mean"], c["pq_mean_over_records"]], w, color=COL[R[n]["v"]], alpha=0.5 + 0.5 * (k % 3) / 3, label=n)
for j, (lab, d) in enumerate((("SQ", hsq), ("RQ", hrq), ("PQ", hq))):
    ax.plot([j - 0.05, j + 0.85], [d["50"]] * 2, color="k", ls="--", lw=1.2)
ax.text(2.88, hq["50"] + 0.01, "human-vs-human median", fontsize=8, ha="right")
ax.set_xticks(np.arange(3) + 0.4); ax.set_xticklabels(["SQ (mask quality of matched instances)", "RQ (matching / detection quality)", "PQ = SQ x RQ"]); ax.legend(fontsize=7, ncol=4, loc="upper center")
ax.set_ylim(0, 0.9); ax.grid(alpha=0.3, axis="y"); ax.set_title("Post 2. Which factor limits the score (best checkpoint of each run)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post2_pq_decomposition.png"), dpi=130); plt.close(fig)

# 3 ---------------------------------------------------------------------------------------------- per-record PQ on shared records
shared = set.intersection(*[set(R[n]["ids"]) for n in names])
fig, ax = plt.subplots(figsize=(10, 4.6))
data, labs, cols = [], [], []
for n in names:
    m = dict(zip(R[n]["ids"], R[n]["pq"])); data.append([m[i] for i in sorted(shared)]); labs.append(n); cols.append(COL[R[n]["v"]])
bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.6)
for q, ls in (("10", ":"), ("50", "--"), ("90", ":")):
    ax.axhline(hq[q], color="gray", ls=ls, lw=1); ax.text(len(names) + 0.45, hq[q], f"human p{q}", fontsize=7, va="center", color="gray")
ax.set_xticklabels(labs, rotation=30, fontsize=8); ax.set_ylabel("per-image PQ"); ax.grid(alpha=0.3, axis="y")
ax.set_title(f"Post 3. Per-image PQ on the {len(shared)} monitoring images shared by all runs (best checkpoints)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post3_per_record_pq.png"), dpi=130); plt.close(fig)

# 4 ---------------------------------------------------------------------------------------------- instance error anatomy
fig, ax = plt.subplots(figsize=(10, 4.8))
cats = [("tp", "matched (IoU > 0.5)", "#59a14f"), ("fp", "extra predicted instances", "#e15759"), ("fn", "missed true instances", "#4e79a7"), ("gt_split_total", "true instances split in pieces", "#f28e2b"), ("pred_merge_total", "predictions merging 2+ true", "#b07aa1")]
w = 0.16
for j, (key, lab, col) in enumerate(cats):
    ax.bar(np.arange(len(names)) + (j - 2) * w, [R[n]["c"][key] / R[n]["c"]["gt_instances"] for n in names], w, color=col, label=lab)
ax.axhline(1.0, color="k", lw=0.6)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_ylabel("count per true instance"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3, axis="y")
ax.set_title("Post 4. Anatomy of the instance errors at the best checkpoint (normalised by the true instances of each subset)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post4_instance_error_anatomy.png"), dpi=130); plt.close(fig)

# 5 ---------------------------------------------------------------------------------------------- recall by size
fig, ax = plt.subplots(figsize=(9, 4.6))
bins = ["300-1000 px", "1000-3000 px", ">=3000 px"]
for i, v in enumerate(vs):
    rows = [R[n]["c"]["recall_by_size"][1:] for n in names if R[n]["v"] == v]
    rec = np.array([[b["recall"] for b in r] for r in rows], dtype=float); cnt = np.array([[b["n"] for b in r] for r in rows])
    ax.bar(np.arange(3) + (i - (len(vs) - 1) / 2) * 0.2, rec.mean(0), 0.2, color=COL[v], label=NAME[v], yerr=[rec.mean(0) - rec.min(0), rec.max(0) - rec.mean(0)], capsize=2)
    for j in range(3):
        ax.text(j + (i - (len(vs) - 1) / 2) * 0.2, 0.02, f"n={int(cnt[:, j].mean())}", ha="center", fontsize=6, color="white")
ax.set_xticks(range(3)); ax.set_xticklabels(bins); ax.set_ylim(0, 1); ax.set_ylabel("share of true instances matched (IoU > 0.5)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.set_title("Post 5. Recall by true-instance area (bars: seed mean, whiskers: min-max; n = instances per bin)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post5_recall_by_size.png"), dpi=130); plt.close(fig)
print("wrote", sorted(f for f in os.listdir(args.out) if f.startswith("post")), "shared records:", len(shared))

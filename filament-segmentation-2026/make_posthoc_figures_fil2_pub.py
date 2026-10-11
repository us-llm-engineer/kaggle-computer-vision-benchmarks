"""Exactly five post-training figures for loop 2, drawn from the result tables (no model is run).
usage: python3 make_posthoc_figures_fil2_pub.py --results results/loop2 --out figures/loop2 [--human-pq 0.359 --target 0.55]
  post1_offline_vs_public.png   CHOOSE PQ (80 records, one fixed instance setting) of every candidate, with the two public scores
  post2_sq_rq.png               SQ against RQ per candidate with iso-PQ curves: where the PQ gain comes from
  post3_error_counts.png        TP / FP / FN per candidate: ensembling removes false positives at a small cost in true positives
  post4_ensemble_gain.png       PQ against the number of averaged models, single models against their ensembles
  post5_selection_ladder.png    the same configuration measured on 16 monitoring records, 80 CHOOSE records and the public test set"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/loop2")
ap.add_argument("--out", default="figures/loop2")
ap.add_argument("--human-pq", type=float, default=0.359, help="median PQ between annotator copies of one image (data investigation)")
ap.add_argument("--target", type=float, default=0.55)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
C = list(csv.DictReader(open(os.path.join(args.results, "loop2_candidates.csv"))))
RUNS = {r["run"]: r for r in csv.DictReader(open(os.path.join(args.results, "loop2_runs.csv")))}
for c in C:
    for k in ("pq", "sq", "rq"):
        c[k] = float(c[k])
    for k in ("tp", "fp", "fn", "n_models"):
        c[k] = int(c[k])
    c["public"] = float(c["public_score"]) if c["public_score"] else None
C.sort(key=lambda c: c["pq"])
names = [c["source"] for c in C]
x = np.arange(len(C))
col = ["#e31a1c" if c["public"] is not None else ("#1f78b4" if c["n_models"] > 1 else "#999999") for c in C]

# post 1 ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x, [c["pq"] for c in C], color=col)
for i, c in enumerate(C):
    ax.text(i, c["pq"] + 0.004, f"{c['pq']:.3f}", ha="center", fontsize=7)
    if c["public"] is not None:
        ax.plot(i, c["public"], "kD", ms=7); ax.text(i + 0.15, c["public"] - 0.012, f"public {c['public']:.2f}", fontsize=7)
ax.axhline(args.human_pq, color="grey", ls="--", lw=1); ax.text(-0.4, args.human_pq + 0.004, "median PQ between two annotator copies", fontsize=7, color="grey")
ax.axhline(args.target, color="g", ls=":", lw=1.2); ax.text(-0.4, args.target + 0.004, f"target {args.target}", fontsize=7, color="g")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8); ax.set_ylim(0, 0.6); ax.set_ylabel("PQ")
ax.set_title("Post 1. CHOOSE PQ per candidate (grey single model, blue ensemble, red submitted; diamonds = public leaderboard score)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post1_offline_vs_public.png"), dpi=130); plt.close(fig)

# post 2 ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
rq = np.linspace(0.54, 0.66, 50)
for p in (0.37, 0.39, 0.41, 0.43):
    ax.plot(rq, p / rq, color="#dddddd", lw=1); ax.text(p / 0.6785, 0.6785, f"PQ {p}", fontsize=7, color="grey", ha="center", va="top")
for c, k in zip(C, col):
    ax.plot(c["rq"], c["sq"], "o", color=k, ms=4 + 2 * c["n_models"]); ax.annotate(c["source"], (c["rq"], c["sq"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("RQ (recognition: F1 of matched instances)"); ax.set_ylabel("SQ (mean IoU of matched instances)"); ax.set_ylim(0.65, 0.68); ax.grid(alpha=0.3)
ax.set_title("Post 2. SQ is flat (0.66 to 0.68); every PQ gain is an RQ gain\n(marker size = number of averaged models)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post2_sq_rq.png"), dpi=130); plt.close(fig)

# post 3 ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6))
w = 0.27
ax.bar(x - w, [c["tp"] for c in C], w, color="#33a02c", label="true positives")
ax.bar(x, [c["fp"] for c in C], w, color="#e31a1c", label="false positives")
ax.bar(x + w, [c["fn"] for c in C], w, color="#1f78b4", label="false negatives (missed)")
gt = C[0]["tp"] + C[0]["fn"]
ax.axhline(gt, color="k", ls=":", lw=1); ax.text(-0.4, gt + 6, f"{gt} annotated instances", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8); ax.set_ylabel("count over 80 CHOOSE records"); ax.legend(fontsize=8)
ax.set_title("Post 3. Instance errors: false positives dominate single models and fall by about half in the ensembles", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post3_error_counts.png"), dpi=130); plt.close(fig)

# post 4 ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for c, k in zip(C, col):
    ax.plot(c["n_models"], c["pq"], "o", color=k, ms=8); ax.annotate(c["source"], (c["n_models"], c["pq"]), fontsize=7, xytext=(5, -3), textcoords="offset points")
single = [c["pq"] for c in C if c["n_models"] == 1]
ax.axhspan(min(single), max(single), color="#eeeeee", zorder=0); ax.text(5.2, max(single) + 0.002, "range of single models", fontsize=7)
ax.set_xlabel("number of averaged probability maps"); ax.set_ylabel("CHOOSE PQ (80 records)"); ax.set_xticks([1, 3, 4, 6]); ax.grid(alpha=0.3)
ax.set_title("Post 4. Averaging models is worth +0.03 to +0.05 PQ; architecture choice within ResNet34 variants is worth about 0.01", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post4_ensemble_gain.png"), dpi=130); plt.close(fig)

# post 5 ---------------------------------------------------------------------------------------------------------
by = {c["source"]: c for c in C}
ladders = [("r34_s0 best checkpoint alone", [float(RUNS["r34_s0"]["best_smoothed_pq_16"]), by["r34best"]["pq"], by["r34best"]["public"]]),
           ("V0 x3 + r34e_s0 (submitted ensemble)", [None, by["ens_v0_r34e"]["pq"], by["ens_v0_r34e"]["public"]])]
fig, ax = plt.subplots(figsize=(8, 5))
stages = ["16 monitoring records\n(checkpoint selection)", "80 CHOOSE records\n(candidate selection)", "public test set\n(leaderboard)"]
for (lab, v), k in zip(ladders, ("#1f78b4", "#e31a1c")):
    xs = [i for i in range(3) if v[i] is not None]
    ax.plot(xs, [v[i] for i in xs], "o-", color=k, label=lab)
    for i in xs:
        ax.text(i + 0.05, v[i] + 0.004, f"{v[i]:.3f}", fontsize=8, color=k)
ax.axhline(args.human_pq, color="grey", ls="--", lw=1); ax.text(0, args.human_pq + 0.003, "median PQ between two annotator copies", fontsize=7, color="grey")
ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=8); ax.set_ylabel("PQ (offline) / public score"); ax.set_ylim(0.28, 0.44); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("Post 5. The same configuration scores lower at every wider evaluation", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "post5_selection_ladder.png"), dpi=130); plt.close(fig)
print("wrote", sorted(f for f in os.listdir(args.out) if f.startswith("post")))

"""Out-of-sample round: exactly nine figures. Six result figures from results/oos_results.json (eval_oos.py) and three in-training figures from the
per-epoch histories out/<variant>_s<seed>_history.json. usage: python3 make_figures_oos.py [--results results/oos_results.json] [--seed 0]"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fn_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(C.OUT, "..", "results", "oos_results.json")); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=os.environ.get("FN_FIG", os.path.join(C.OUT, "..", "figures_oos")))
args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)
R = json.load(open(args.results)); V = list(R)
NAME = {"bce": "V0 control (BCE)", "vmf": "V1 hypersphere outliers", "radius": "V2 radius head", "evid": "V3 evidential sum", "comb": "V4 learned combiner"}
COL = {"bce": "#555555", "vmf": "#1f77b4", "radius": "#d62728", "evid": "#2ca02c", "comb": "#9467bd"}
FLOOR = 0.55
hist = {}
for v in V:
    path = os.path.join(C.OUT, f"{v}_s{args.seed}_history.json")
    if os.path.exists(path): hist[v] = json.load(open(path))
save = lambda fig, n: (fig.savefig(os.path.join(args.outdir, n), dpi=130, bbox_inches="tight"), plt.close(fig))
lab = lambda v: NAME.get(v, v)

# 1. score against the 0.55 floor, halves stacked
fig, ax = plt.subplots(figsize=(8, 4.6)); x = np.arange(len(V))
sa = [0.5 * R[v]["ensemble"]["sauc_all"] for v in V]; mp = [0.5 * R[v]["ensemble"]["map_all_tail"] for v in V]
ax.bar(x, sa, color="#8da0cb", label="0.5 x sAUC"); ax.bar(x, mp, bottom=sa, color="#fc8d62", label="0.5 x all-image MAP@20 (with tail)")
ax.errorbar(x, [R[v]["ensemble"]["score_tail"] for v in V], yerr=[R[v]["band_score_tail"]["half_range"] for v in V], fmt="k_", capsize=6, label="seed half-range")
ax.axhline(FLOOR, color="k", ls="--", lw=1.2); ax.text(len(V) - 0.5, FLOOR + 0.006, "floor 0.55", ha="right")
ax.set_xticks(x); ax.set_xticklabels([lab(v) for v in V], rotation=12); ax.set_ylabel("competition-style score on REPORT"); ax.set_title("Score against the 0.55 floor (3-seed ensemble)"); ax.legend(loc="upper left", fontsize=8)
save(fig, "fig1_score_vs_floor.png")

# 2. sAUC by novel-image type
fig, ax = plt.subplots(figsize=(8, 4.4)); w = 0.26
for j, (k, t) in enumerate((("sauc_all", "all novel"), ("sauc_pure", "pure novel"), ("sauc_mixed", "mixed seen+novel"))):
    vals = [R[v]["ensemble"][k] if R[v]["ensemble"][k] is not None else np.nan for v in V]
    ax.bar(x + (j - 1) * w, vals, w, label=t)
    for i, v in enumerate(V):
        s = [r[k] for r in R[v]["per_seed"] if r.get(k) is not None]; ax.scatter(np.full(len(s), i + (j - 1) * w), s, color="k", s=8, zorder=3)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([lab(v) for v in V], rotation=12); ax.set_ylabel("sAUC = 2 AUC - 1 (REPORT)")
ax.set_title("Novelty detection by image type (dots: single seeds)"); ax.legend(fontsize=8)
save(fig, "fig2_sauc_by_group.png")

# 3. ROC curves
fig, ax = plt.subplots(figsize=(5.2, 5))
for v in V: ax.plot(R[v]["roc"]["fpr"], R[v]["roc"]["tpr"], color=COL.get(v), label=f"{lab(v)}: sAUC {R[v]['ensemble']['sauc_all']:.3f} ({R[v]['chosen_score']})")
ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("false positive rate (seen images flagged novel)"); ax.set_ylabel("true positive rate (novel images)"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("ROC of the selected score on REPORT")
save(fig, "fig3_roc.png")

# 4. MAP@20 seen / novel / all, with and without the tail
fig, ax = plt.subplots(figsize=(9, 4.4)); keys = [("map_seen", "seen images"), ("map_novel", "novel images"), ("map_novel_tail", "novel + tail"), ("map_all", "all images"), ("map_all_tail", "all + tail")]
w = 0.16
for j, (k, t) in enumerate(keys): ax.bar(x + (j - 2) * w, [R[v]["ensemble"][k] for v in V], w, label=t)
ax.set_xticks(x); ax.set_xticklabels([lab(v) for v in V], rotation=12); ax.set_ylabel("MAP@20 (REPORT + ID validation)"); ax.set_title("Ranking quality: seen vs novel images and the ranking tail"); ax.legend(fontsize=8, ncol=2)
save(fig, "fig4_map_tail.png")

# 5. per held-out category / supercategory
fig, ax = plt.subplots(figsize=(8.5, 4.4)); cats = {}
for v in V:
    for c in R[v].get("per_category", []): cats.setdefault((c["category"], c["supercat"], c["n"]), {})[v] = c["sauc"]
labels = [f"{k[1]} (cat {k[0]}, n={k[2]})" for k in cats]; xx = np.arange(len(cats)); w = 0.8 / max(len(V), 1)
for j, v in enumerate(V): ax.bar(xx + (j - len(V) / 2 + 0.5) * w, [cats[k].get(v, np.nan) for k in cats], w, color=COL.get(v), label=lab(v))
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8); ax.set_ylabel("sAUC against ID validation"); ax.set_title("Novelty detection per held-out REPORT category"); ax.legend(fontsize=7)
save(fig, "fig5_per_category.png")

# 6. selection optimism: SELECT sAUC against REPORT sAUC
fig, ax = plt.subplots(figsize=(5.4, 5))
for v in V:
    xs = [r["sauc_select"] for r in R[v]["per_seed"]]; ys = [r["sauc_all"] for r in R[v]["per_seed"]]; ax.scatter(xs, ys, color=COL.get(v), label=lab(v))
lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]; ax.plot(lim, lim, "k:", lw=0.8)
ax.set_xlabel("sAUC on SELECT (used for choosing)"); ax.set_ylabel("sAUC on REPORT (read once)"); ax.set_title("Selection optimism: points below the line"); ax.legend(fontsize=7)
save(fig, "fig6_selection_optimism.png")

# 7. in-training: score distributions and sAUC by group
H = [v for v in V if v in hist]
fig, axes = plt.subplots(len(H), 4, figsize=(15, 3.1 * max(len(H), 1)), squeeze=False)
for i, v in enumerate(H):
    h = hist[v]["history"]; picks = [0, len(h) // 2, int(np.argmax([r["crit_smooth"] for r in h]))]
    for j, e in enumerate(picks):
        r = h[e]; b = np.array(r["hist_bins"]); c = (b[:-1] + b[1:]) / 2
        axes[i, j].step(c, np.array(r["hist_in"]) / max(sum(r["hist_in"]), 1), where="mid", label="ID validation"); axes[i, j].step(c, np.array(r["hist_novel"]) / max(sum(r["hist_novel"]), 1), where="mid", label="SELECT novel")
        axes[i, j].set_title(f"{lab(v)}: epoch {r['epoch']} ({r['best_score_key']})", fontsize=8); axes[i, j].legend(fontsize=6)
    ep = [r["epoch"] for r in h]; ax = axes[i, 3]
    for k, t in (("sauc_select_pure", "SELECT pure"), ("sauc_select_mixed", "SELECT mixed"), ("sauc_pool", "POOL"), ("sauc_train_vs_val", "train vs val (memorisation)")):
        ax.plot(ep, [np.nan if r["interp"][k] is None else r["interp"][k] for r in h], label=t)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{lab(v)}: novelty sAUC by group", fontsize=8); ax.set_xlabel("epoch"); ax.legend(fontsize=6)
fig.suptitle("In training: score distributions of seen vs novel images and sAUC by image group", y=1.0)
save(fig, "fig7_training_distributions.png")

# 8. in-training: variant geometry, unseen-output behaviour and prototype distances
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
a = axes[0, 0]
for v in H:
    h = hist[v]["history"]; a.plot([r["epoch"] for r in h], [r["interp"]["group"]["select"]["unseen_logit_mean"] for r in h], color=COL.get(v), label=lab(v))
a.set_title("mean logit of never-positive outputs\n(SELECT images)", fontsize=9); a.legend(fontsize=6)
a = axes[0, 1]
for v in H:
    h = hist[v]["history"]; a.plot([r["epoch"] for r in h], [r["interp"]["proto"]["maxcos_select"] - r["interp"]["proto"]["maxcos_val_in"] for r in h], color=COL.get(v), label=lab(v))
a.set_title("max cosine to a seen prototype:\nSELECT minus ID validation", fontsize=9)
a = axes[0, 2]
for v in H:
    h = hist[v]["history"]; a.plot([r["epoch"] for r in h], [r["interp"]["mixed_seen_label_recall"] for r in h], color=COL.get(v), label=lab(v))
a.set_title("seen-label probability on\nmixed SELECT images", fontsize=9)
a = axes[1, 0]
if "vmf" in hist:
    h = hist["vmf"]["history"]; ep = [r["epoch"] for r in h]
    a.plot(ep, [r["interp"]["virtual_vs_real_novelty"]["val_in"] for r in h], label="real ID validation"); a.plot(ep, [r["interp"]["virtual_vs_real_novelty"]["select"] for r in h], label="real SELECT novel")
    a.plot(ep, [np.nan if r["interp"]["virtual_outliers"] is None else r["interp"]["virtual_outliers"]["novelty_mean"] for r in h], label="virtual outliers"); a.legend(fontsize=6)
a.set_title("V1: joint-energy novelty of virtual\noutliers vs real images", fontsize=9)
a = axes[1, 1]
if "radius" in hist:
    h = hist["radius"]["history"]; ep = [r["epoch"] for r in h]
    for k in ("val_in", "select", "pool"): a.plot(ep, [np.nan if r["interp"]["radius"] is None else r["interp"]["radius"][k] for r in h], label=f"radius {k}")
    a.plot(ep, [r["geometry"]["radius_mix"] if r["geometry"]["radius_mix"] is not None else np.nan for r in h], label="radius of mixtures"); a.legend(fontsize=6)
a.set_title("V2: feature radius around the ID centre", fontsize=9)
a = axes[1, 2]
if "evid" in hist:
    h = hist["evid"]["history"]; ep = [r["epoch"] for r in h]; a.plot(ep, [r["geometry"]["alpha_mean"] for r in h], label="mean positive evidence"); a.plot(ep, [r["geometry"]["beta_mean"] for r in h], label="mean negative evidence"); a.legend(fontsize=6)
a.set_title("V3: Beta evidence during training", fontsize=9)
for a in axes.ravel(): a.set_xlabel("epoch", fontsize=8)
fig.subplots_adjust(hspace=0.45, wspace=0.3); fig.suptitle("In training: how the held-back groups and the outlier handling change the model", y=0.98)
save(fig, "fig8_training_geometry.png")

# 9. in-training: optimisation dynamics, MAP by group, throughput
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for v in H:
    h = hist[v]["history"]; ep = [r["epoch"] for r in h]
    axes[0, 0].plot(ep, [r["train_loss"] for r in h], color=COL.get(v), label=lab(v)); axes[0, 1].plot(ep, [r["crit_smooth"] for r in h], color=COL.get(v))
    axes[0, 2].plot(ep, [r["lr"] for r in h], color=COL.get(v)); axes[1, 1].plot(ep, [r["interp"]["map"]["val_in"] for r in h], color=COL.get(v)); axes[1, 1].plot(ep, [r["interp"]["map"]["train"] for r in h], color=COL.get(v), ls=":")
    axes[1, 2].plot(ep, [r["images_per_second"] for r in h], color=COL.get(v))
    g = h[-1]["grad_norm_by_group"]; axes[1, 0].bar(np.arange(len(g)) + 0.2 * H.index(v), list(g.values()), 0.2, color=COL.get(v))
axes[0, 0].set_title("training loss", fontsize=9); axes[0, 0].legend(fontsize=6); axes[0, 1].set_title("smoothed selection criterion (early-stopping signal)", fontsize=9)
axes[0, 2].set_title("learning rate (warm-up, plateau cuts)", fontsize=9); axes[0, 2].set_yscale("log")
axes[1, 0].set_title("final-epoch gradient norm per layer group", fontsize=9); axes[1, 0].set_xticks(np.arange(len(g))); axes[1, 0].set_xticklabels(list(g), rotation=70, fontsize=6)
axes[1, 1].set_title("MAP@20 on ID validation (solid) and train subset (dotted)", fontsize=9); axes[1, 2].set_title("throughput (images/s)", fontsize=9)
for a in axes.ravel(): a.tick_params(labelsize=7)
fig.suptitle("In training: optimisation dynamics, memorisation and throughput", y=1.0)
save(fig, "fig9_training_dynamics.png")
print("figures:", sorted(os.listdir(args.outdir)))

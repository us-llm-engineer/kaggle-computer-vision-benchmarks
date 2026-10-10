"""Draws exactly nine figures (six result figures and three in-training interpretation figures) from the statistics files and saved predictions written by the training scripts, and writes the results table and the
submission files (not submitted anywhere).

  fig1  training dynamics: loss, MAP@20 with the generalisation gap, sAUC for three scores, learning rate with gradient norm, and per-epoch
        heatmaps of how the max-probability histograms of in-distribution and novel validation images separate
  fig2  long-tail view: per-category average precision against training frequency, bubble size = validation positives, colour = supercategory
  fig3  out-of-sample scores: ridgeline densities of 1 - max probability (in-distribution vs novel) and ROC curves for two scores
  fig4  t-SNE of the learned features of both deep models, coloured by supercategory, novel images marked
  fig5  Grad-CAM of the CNN on the best and worst validation images with the ground-truth boxes drawn on the same pictures
  fig6  clustered heatmap of mean average precision per supercategory and methodology
  fig7  (in training) per-supercategory average-precision trajectories: which groups are learned first, and which never are
  fig8  (in training) layer-wise gradient norm and relative weight-update heatmaps over epochs: where in the network learning happens and when it stalls
  fig9  (in training) reliability diagrams coloured by epoch, calibration error per epoch, and the epoch at which each category is first learned
usage: python3 make_figures.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_curve

import fn_common as C
import fn_models as M

FIG = os.environ.get("FN_FIG", f"{C.ROOT}/figures"); RES = os.environ.get("FN_RESULTS", f"{C.ROOT}/results"); SUBS = os.environ.get("FN_SUBS", f"{C.ROOT}/submissions")
for d in (FIG, RES, SUBS): os.makedirs(d, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
NAMES = {"classical": "1 Classical features + ridge", "cnn": "2 Residual CNN (224x128)", "mil": "3 Tile-MIL CNN (448x256)"}
COL = {"classical": "#8c8c8c", "cnn": "#1f77b4", "mil": "#d95f02"}
meta, sp = C.load_prep(); Yall = np.load(f"{C.PREP}/train_y.npy"); tr = sp["train"]; freq = Yall[tr].sum(0)
sc, sc_names = C.supercat_index(); key = C.category_key()
hist = {k: json.load(open(f"{C.OUT}/{k}_history.json")) for k in NAMES}
val = {k: np.load(f"{C.OUT}/{k}_val.npz") for k in NAMES}
val_idx, novel = val["cnn"]["val_idx"], val["cnn"]["novel"]; Yv = Yall[val_idx]
prob = {k: 1 / (1 + np.exp(-val[k]["logits"].astype(np.float64))) for k in NAMES}
pal = dict(zip(sc_names, sns.color_palette("tab20", len(sc_names))))


def class_ap(k):
    p, inn = prob[k], ~novel; out = {}
    for c in range(C.NCLS):
        if Yv[inn][:, c].sum() > 0: out[c] = average_precision_score(Yv[inn][:, c], p[inn][:, c])
    return out


AP = {k: class_ap(k) for k in NAMES}

# ------------------------------------------------------------------------------------------------------------------ figure 1
fig = plt.figure(figsize=(17, 9.5)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.45); ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
for k in ("cnn", "mil"):
    h = pd.DataFrame(hist[k]["history"]); e = h.epoch
    ax[0].plot(e, h.train_loss, color=COL[k], label=f"{k} train"); ax[0].plot(e, h.val_loss, color=COL[k], ls="--", label=f"{k} validation")
    ax[1].plot(e, h.val_map20, color=COL[k], label=f"{k} validation"); ax[1].plot(e, h.train_map20_eval, color=COL[k], ls=":", label=f"{k} train subset")
    ax[1].fill_between(e, h.val_map20, h.train_map20_eval, color=COL[k], alpha=0.10)
    for s, ls in (("msp", "-"), ("energy", "--"), ("entropy", ":")): ax[2].plot(e, h[f"val_sauc_{s}"], color=COL[k], ls=ls, label=f"{k} {s}")
    ax[3].plot(e, h.lr, color=COL[k], label=f"{k} learning rate")
    twin = ax[3].twinx() if k == "cnn" else twin; twin.plot(e, h.grad_norm_mean, color=COL[k], ls="--", alpha=0.8); twin.set_yscale("log")
ax[1].axhline(hist["classical"]["best"]["map20"], color=COL["classical"], lw=1.2, label="classical (best)"); ax[2].axhline(hist["classical"]["best"]["sauc_msp"], color=COL["classical"], lw=1.2, label="classical msp")
ax[0].set(title="a  Loss (balanced BCE)", xlabel="epoch"); ax[1].set(title="b  MAP@20 and the generalisation gap (shaded)", xlabel="epoch"); ax[2].set(title="c  Novel-image detection, sAUC by score", xlabel="epoch")
ax[3].set(title="d  Learning rate (solid) and gradient norm (dashed)", xlabel="epoch"); twin.set_ylabel("mean gradient norm")
for a in ax[:4]: a.legend(fontsize=7.5, ncol=2, frameon=True)
for a, k in zip(ax[4:], ("cnn", "mil")):
    h = hist[k]["history"]; A = np.array([np.array(r["msp_hist_in"]) / max(sum(r["msp_hist_in"]), 1) - np.array(r["msp_hist_novel"]) / max(sum(r["msp_hist_novel"]), 1) for r in h]).T
    im = a.imshow(A, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-np.abs(A).max(), vmax=np.abs(A).max(), extent=[0.5, len(h) + 0.5, 0, 1])
    a.set(title=f"{'e' if k == 'cnn' else 'f'}  {k}: max-probability histogram, in-dist. minus novel", xlabel="epoch", ylabel="max-probability bin"); plt.colorbar(im, ax=a, pad=0.01, label="share(in) - share(novel)")
fig.suptitle("Figure 1  Training dynamics of the two deep methodologies", fontsize=14, y=0.995); fig.savefig(f"{FIG}/fig1_training_dynamics.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 2
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
for a, k in zip(axes, NAMES):
    cs = np.array(sorted(AP[k])); x, y = freq[cs], np.array([AP[k][c] for c in cs]); npos = Yv[~novel][:, cs].sum(0)
    a.scatter(np.maximum(x, 0.7), y, s=18 + 14 * np.sqrt(npos), c=[pal[sc_names[sc[c]]] for c in cs], alpha=0.75, edgecolor="white", lw=0.5)
    order = np.argsort(x); bins = np.array_split(order, 6)
    a.plot([np.median(np.maximum(x[b], 0.7)) for b in bins], [np.median(y[b]) for b in bins], color="black", lw=2, marker="o", label="binned median")
    a.set(xscale="log", title=f"{NAMES[k]}\nmacro AP {y.mean():.3f} over {len(cs)} categories", xlabel="training images with the category (log)"); a.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("average precision on in-distribution validation")
axes[2].legend(handles=[Line2D([], [], marker="o", ls="", color=pal[n], label=n) for n in sc_names], fontsize=6.5, ncol=2, loc="lower right", title="supercategory", title_fontsize=7)
fig.suptitle("Figure 2  Long tail: rare categories drive the error (bubble size = validation positives)", fontsize=14); fig.savefig(f"{FIG}/fig2_long_tail.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 3
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8)); grid = np.linspace(0, 1, 300)
for row, k in enumerate(NAMES):
    s = C.ood_scores(val[k]["logits"])["msp"]
    for flag, color, lab in ((False, "#2b83ba", "in-distribution"), (True, "#d7191c", "novel")):
        d = gaussian_kde(s[novel == flag], bw_method=0.25)(grid); d = d / d.max() * 0.85
        a1.fill_between(grid, row, row + d, color=color, alpha=0.55, label=lab if row == 0 else None); a1.plot(grid, row + d, color=color, lw=1)
    a1.text(0.0, row + 0.88, NAMES[k], fontsize=9)
a1.set(title="a  Out-of-sample score = 1 - max probability", xlabel="score (higher = more novel)", yticks=[]); a1.legend(loc="upper center", ncol=2)
for k in NAMES:
    for s, ls in (("msp", "-"), ("energy", "--")):
        fpr, tpr, _ = roc_curve(novel.astype(int), C.ood_scores(val[k]["logits"])[s]); a2.plot(fpr, tpr, color=COL[k], ls=ls, label=f"{k} {s}: sAUC {2 * float(np.trapz(tpr, fpr) if not hasattr(np, 'trapezoid') else np.trapezoid(tpr, fpr)) - 1:.3f}")
a2.plot([0, 1], [0, 1], color="grey", lw=0.8); a2.set(title="b  ROC of novel-vs-in-distribution detection", xlabel="false positive rate", ylabel="true positive rate"); a2.legend(fontsize=8)
fig.suptitle("Figure 3  Detecting images that show unseen categories (validation proxy for out-of-sample)", fontsize=14); fig.savefig(f"{FIG}/fig3_out_of_sample.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 4
fig, axes = plt.subplots(1, 2, figsize=(15, 6.6)); rare = np.where(Yv > 0, freq[None, :] + 1e-9, np.inf).argmin(1)
for a, k in zip(axes, ("cnn", "mil")):
    f = val[k]["feats"].astype(np.float32); z = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(PCA(40, random_state=0).fit_transform(f))
    cols = np.array([pal[sc_names[sc[c]]] for c in rare]); a.scatter(z[~novel, 0], z[~novel, 1], c=cols[~novel], s=14, alpha=0.8, linewidth=0)
    a.scatter(z[novel, 0], z[novel, 1], c="black", marker="x", s=18, lw=0.9, label="novel (held-out categories)"); a.set(title=NAMES[k], xticks=[], yticks=[]); a.legend(loc="lower left", fontsize=8)
axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=pal[n], label=n) for n in sc_names] + [Line2D([], [], marker="x", ls="", color="black", label="novel")], fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.suptitle("Figure 4  t-SNE of validation features (colour = supercategory of the rarest label)", fontsize=14); fig.savefig(f"{FIG}/fig4_embedding.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 5
dev = os.environ.get("FN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"); net = M.build("cnn").to(dev); net.load_state_dict(torch.load(f"{C.CKPT}/cnn_best.pt", map_location=dev, weights_only=False)["model"]); net.eval()
X = np.load(f"{C.PREP}/train_x224.npy", mmap_mode="r"); ids = np.load(f"{C.PREP}/train_ids.npy"); coco = json.load(open(f"{C.DATA}/object_detection/train.json"))
by_name = {C.stem(i["file_name"]): i for i in coco["images"]}; boxes = {}
for an in coco["annotations"]: boxes.setdefault(an["image_id"], []).append(an["bbox"])
inn = np.where(~novel)[0]; p = prob["cnn"][inn]; per = np.array([C.ap_list((np.argsort(-p[i])[:20] + 1).tolist(), set((np.where(Yv[inn][i])[0] + 1).tolist())) for i in range(len(inn))])
order = np.argsort(per); pick = np.r_[order[-4:][::-1], order[:4]]
mean = torch.tensor(meta["channel_mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(meta["channel_std"], device=dev).view(1, 3, 1, 1)
fig, axes = plt.subplots(2, 4, figsize=(17, 5.8)); names = dict(zip(key.id.astype(int), key.name))
for a, j in zip(axes.ravel(), pick):
    gi = val_idx[inn[j]]; img = np.asarray(X[gi]); x = ((torch.from_numpy(img.copy()).to(dev).permute(2, 0, 1)[None].float() / 255 - mean) / std).requires_grad_(True)
    feat = net.trunk(x); feat.retain_grad(); logit = net.fc(torch.cat([feat.mean((2, 3)), feat.amax((2, 3))], 1)); c = int(logit.argmax()); logit[0, c].backward()
    w = feat.grad.mean((2, 3), keepdim=True); cam = F.relu((w * feat).sum(1, keepdim=True)); cam = F.interpolate(cam, size=img.shape[:2], mode="bilinear")[0, 0].detach().cpu().numpy(); cam = cam / (cam.max() + 1e-9)
    a.imshow(img); a.imshow(cam, cmap="jet", alpha=0.38); info = by_name[C.stem(ids[gi])]
    for bx, by, bw, bh in boxes.get(info["id"], []):
        sx, sy = img.shape[1] / info["width"], img.shape[0] / info["height"]; a.add_patch(plt.Rectangle((bx * sx, by * sy), bw * sx, bh * sy, fill=False, ec="lime", lw=1.4))
    truth = ", ".join(names[c_ + 1] for c_ in np.where(Yv[inn][j])[0][:2]); a.set_title(f"AP {per[j]:.2f} | pred {names[c + 1]} ({float(torch.sigmoid(logit[0, c])):.2f})\ntrue: {truth}", fontsize=7.5); a.axis("off")
fig.suptitle("Figure 5  Grad-CAM of the residual CNN: four best (top row) and four worst (bottom row) validation images, ground-truth boxes in green", fontsize=13)
fig.savefig(f"{FIG}/fig5_gradcam.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 6
M6 = pd.DataFrame({NAMES[k]: {sc_names[s]: np.mean([AP[k][c] for c in AP[k] if sc[c] == s]) for s in range(len(sc_names)) if any(sc[c] == s for c in AP[k])} for k in NAMES})
counts = {sc_names[s]: sum(1 for c in AP["cnn"] if sc[c] == s) for s in range(len(sc_names))}; M6.index = [f"{n} ({counts[n]})" for n in M6.index]
g = sns.clustermap(M6, annot=True, fmt=".2f", cmap="viridis", col_cluster=False, figsize=(9, 8), cbar_kws={"label": "mean average precision"}, dendrogram_ratio=(0.18, 0.02), linewidths=0.5)
g.figure.suptitle("Figure 6  Mean AP per supercategory (number of validation categories in brackets), rows clustered", y=1.02, fontsize=12); g.savefig(f"{FIG}/fig6_supercategory_heatmap.png", dpi=150, bbox_inches="tight"); plt.close("all")


# ------------------------------------------------------------------------------------------------------------------ figure 7
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
for a, k in zip(axes, ("cnn", "mil")):
    h = hist[k]["history"]; ap_e = np.array([[np.nan if v is None else v for v in r["class_ap"]] for r in h]); ep = [r["epoch"] for r in h]
    for si, n in enumerate(sc_names):
        cols = [c for c in range(C.NCLS) if sc[c] == si and not np.isnan(ap_e[:, c]).all()]
        if cols: a.plot(ep, np.nanmean(ap_e[:, cols], 1), color=pal[n], lw=1.8, label=f"{n} ({len(cols)})")
    a.plot(ep, np.nanmean(ap_e, 1), color="black", lw=2.6, ls="--", label="all categories"); a.set(title=NAMES[k], xlabel="epoch")
axes[0].set_ylabel("mean AP of the supercategory (validation)"); axes[1].legend(fontsize=6.5, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="supercategory (categories)", title_fontsize=7)
fig.suptitle("Figure 7  In training: which supercategories are learned, and when", fontsize=14); fig.savefig(f"{FIG}/fig7_supercategory_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 8
fig, axes = plt.subplots(2, 2, figsize=(15, 8.6))
for col, k in enumerate(("cnn", "mil")):
    h = hist[k]["history"]; gn = h[0]["group_names"]
    for row, (key_, lab, cmap) in enumerate((("update_ratio_by_group", "relative weight update  ||Δw|| / ||w||  (log10)", "mako"), ("grad_norm_by_group", "mean gradient norm per group (log10)", "rocket"))):
        A = np.log10(np.array([r[key_] for r in h]).T + 1e-12); a = axes[row, col]
        im = a.imshow(A, aspect="auto", origin="lower", cmap=sns.color_palette(cmap, as_cmap=True), extent=[0.5, len(h) + 0.5, -0.5, len(gn) - 0.5])
        a.set(yticks=range(len(gn)), yticklabels=gn, xlabel="epoch", title=f"{NAMES[k]}\n{lab}"); a.xaxis.set_major_locator(MaxNLocator(integer=True)); plt.colorbar(im, ax=a, pad=0.01)
fig.suptitle("Figure 8  In training: where in the network the learning happens (stem at the bottom, head at the top)", fontsize=14); fig.tight_layout(); fig.savefig(f"{FIG}/fig8_layerwise_learning.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------ figure 9
fig, axes = plt.subplots(1, 4, figsize=(21, 5.2)); cm = plt.get_cmap("viridis")
for a, k in zip(axes[:2], ("cnn", "mil")):
    h = hist[k]["history"]
    for r in h:
        n_, pos_, conf_ = np.array(r["rel_counts"]), np.array(r["rel_positives"]), np.array(r["rel_mean_conf"]); ok = n_ >= 20
        a.plot(conf_[ok] / n_[ok], pos_[ok] / n_[ok], color=cm(r["epoch"] / len(h)), lw=1.1, marker="o", ms=2.5)
    a.plot([1e-4, 1], [1e-4, 1], color="black", lw=1, ls="--"); a.set(xscale="log", yscale="log", xlabel="mean predicted probability in the bin", ylabel="observed positive rate", title=f"{k}: reliability (colour = epoch)", ylim=(1e-4, 1.5))
ece = {}
for k in ("cnn", "mil"):
    e_ = []
    for r in hist[k]["history"]:
        n_, pos_, conf_ = np.array(r["rel_counts"]), np.array(r["rel_positives"]), np.array(r["rel_mean_conf"]); e_.append(np.abs(pos_ - conf_).sum() / n_.sum())
    ece[k] = e_; axes[2].plot(range(1, len(e_) + 1), e_, color=COL[k], label=NAMES[k])
axes[2].set(yscale="log", xlabel="epoch", ylabel="calibration error per label entry", title="Calibration error over training"); axes[2].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[2].legend(fontsize=8)
for k in ("cnn", "mil"):
    ap_e = np.array([[np.nan if v is None else v for v in r["class_ap"]] for r in hist[k]["history"]]); first = np.array([next((i + 1 for i, v in enumerate(ap_e[:, c]) if v >= 0.3), np.nan) for c in range(C.NCLS)])
    ok = ~np.isnan(ap_e).all(0); jitter = np.random.RandomState(0).uniform(-0.25, 0.25, C.NCLS)
    axes[3].scatter(np.maximum(freq[ok], 0.7), np.where(np.isnan(first[ok]), len(ap_e) + 1.5, first[ok]) + jitter[ok], s=12, color=COL[k], alpha=0.55, label=NAMES[k])
axes[3].set(xscale="log", xlabel="training images with the category (log)", ylabel="first epoch with validation AP >= 0.3 (top = never)", title="Rare categories are learned late or never"); axes[3].legend(fontsize=8)
fig.suptitle("Figure 9  In training: calibration and learning time", fontsize=14); fig.subplots_adjust(wspace=0.38); fig.savefig(f"{FIG}/fig9_calibration_and_learning_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------- table + submissions
rows = []
for k in NAMES:
    h = hist[k]; best = h["best"] if k == "classical" else max(h["history"], key=lambda r: r["val_proxy_score"]); get = (lambda n: best[n]) if k == "classical" else (lambda n: best[f"val_{n}"])
    rows.append({"methodology": NAMES[k], "best_epoch": best.get("epoch", "-"), "val_MAP@20": get("map20"), "val_sAUC_msp": get("sauc_msp"), "val_sAUC_energy": get("sauc_energy"), "val_sAUC_entropy": get("sauc_entropy"),
                 "proxy_score": get("proxy_score"), "train_MAP@20": best["train_map20"] if k == "classical" else best["train_map20_eval"], "parameters": "closed form" if k == "classical" else h["params"],
                 "macro_AP_categories": float(np.mean(list(AP[k].values())))})
tab = pd.DataFrame(rows).round(4); tab.to_csv(f"{RES}/final_metrics.csv", index=False); print(tab.to_string(index=False))
for k in NAMES:
    t = np.load(f"{C.OUT}/{k}_test.npz", allow_pickle=True); lg = t["logits"]; top = np.argsort(-lg, 1)[:, :20] + 1
    pd.DataFrame({"id": t["ids"], "categories": [" ".join(map(str, r)) for r in top], "osd": C.ood_scores(lg)["msp"]}).to_csv(f"{SUBS}/submission_{k}.csv", index=False)
print("figures:", sorted(os.listdir(FIG)))

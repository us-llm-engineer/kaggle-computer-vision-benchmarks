"""Out-of-sample round, V4: a learned combiner over post-hoc novelty measures (COOD idea) plus a small outlier expert (MARVEL idea) trained on
real held-back categories of our own data (the POOL group) instead of an external auxiliary set.

Everything is computed on the frozen features and logits of a V0 network (train_oos.py bce). The kNN reference set is the training images. The
combiner learns ID-validation images (negatives) against POOL images (positives). To keep evaluation honest the ID-validation negatives and the POOL
positives are cross-fitted: five folds, each fold's scores come from a combiner that never saw that fold, and SELECT / REPORT images are scored by the mean
of the five combiners. SELECT and REPORT images are never used for fitting.
usage: python3 combiner_oos.py --pred out/bce_s0_pred.npz --out out/comb_s0_pred.npz [--k 10] [--no-expert]"""
import argparse

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import fn_common as C

l2n = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def entropy(p):
    q = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
    return -(q * np.log(np.clip(q, 1e-12, None))).sum(1)


class MeasureBank:
    """Fits the reference structures on the training features once, then turns (features, logits) into a (n, m) measure matrix."""

    def __init__(self, Ftr, Ytr, sc_of, k=10, n_pca=64, min_class=12):
        self.k, self.Ytr, self.sc_of = k, Ytr.astype(np.float32), sc_of
        self.pca_nn = PCA(n_components=min(256, Ftr.shape[1] - 1, len(Ftr) - 1)).fit(Ftr)
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(self.pca_nn.transform(Ftr))
        self.pca_g = PCA(n_components=min(n_pca, Ftr.shape[1] - 1, len(Ftr) - 1)).fit(Ftr)
        self.class_pca = {}
        top = Ytr.argmax(1)                                            # class of an image = its first positive category
        for c in np.unique(top):
            m = top == c
            if m.sum() >= min_class: self.class_pca[c] = PCA(n_components=min(8, m.sum() - 1)).fit(Ftr[m])

    def fre(self, F, pca):
        return ((F - pca.inverse_transform(pca.transform(F))) ** 2).sum(1)

    def __call__(self, F, logits):
        p = expit(np.asarray(logits, np.float64)); dist, idx = self.nn.kneighbors(self.pca_nn.transform(F))
        pN = self.Ytr[idx].mean(1)
        top_lin, top_nn = p.argmax(1), pN.argmax(1)
        td = np.where(top_lin == top_nn, 0.0, np.where(self.sc_of[top_lin] == self.sc_of[top_nn], 0.5, 1.0))
        hN = entropy(pN + 1e-9); gfre = self.fre(F, self.pca_g)
        cfre = gfre.copy()
        for c, pca in self.class_pca.items():
            m = top_lin == c
            if m.any(): cfre[m] = self.fre(F[m], pca)
        a = np.abs(F); fe = entropy(a)
        cols = {"tmax": 1 - expit(np.asarray(logits, np.float64) / 2.0).max(1), "max": 1 - p.max(1), "joint_energy": -np.logaddexp(0, logits).sum(1), "entropy": entropy(p),
                "knn_max": 1 - pN.max(1), "lin_knn_max": 1 - ((p + pN) / 2).max(1), "taxon_dist": td, "avg_dist_nn": dist.mean(1), "dist_1nn": dist[:, 0],
                "enwedi_1": dist[:, 0] * (1 + hN), "enwedi_avg": dist.mean(1) * (1 + hN), "knn_entropy": hN, "feature_entropy": fe, "global_fre": gfre, "class_fre": cfre}
        return np.stack(list(cols.values()), 1), list(cols)


def run(pred_path, out_path, k=10, expert=True, seed=0, folds=5):
    P = np.load(pred_path, allow_pickle=True); Y = np.load(f"{C.PREP}/train_y.npy"); sc_of, _ = C.supercat_index()
    F = lambda g: P[f"{g}_feat"].astype(np.float64); L = lambda g: P[f"{g}_logit"].astype(np.float64)
    bank = MeasureBank(F("train"), Y[P["train_idx"]], sc_of, k=k)
    M = {}
    for g in ("val_in", "select", "pool", "report", "eval"): M[g], names = bank(F(g), L(g))
    # positives: POOL and SELECT images (held-back categories); negatives: ID validation. Every image of the three groups gets an out-of-fold score, so SELECT
    # stays usable for choosing tau; REPORT and eval images are scored by the mean of the fold models and never enter a fit.
    scaler = StandardScaler().fit(np.r_[F("val_in"), F("pool"), F("select")])
    fit_g = ("val_in", "pool", "select"); Xi = np.concatenate([M[g] for g in fit_g]); feats_fit = scaler.transform(np.concatenate([F(g) for g in fit_g]))
    y = np.concatenate([np.full(len(M[g]), int(g != "val_in")) for g in fit_g]); sizes = [len(M[g]) for g in fit_g]
    oof = np.zeros(len(y)); out = {g: np.zeros(len(M[g])) for g in ("report", "eval")}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(Xi, y):
        cols_tr, cols_te = Xi[tr_i], Xi[te_i]
        if expert:   # outlier expert: its training-part scores are produced out-of-fold (inner 5-fold) so the forest never sees over-confident in-sample scores
            lr_ = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced")
            e_tr = cross_val_predict(lr_, feats_fit[tr_i], y[tr_i], cv=5, method="decision_function"); lr = lr_.fit(feats_fit[tr_i], y[tr_i])
            cols_tr = np.c_[cols_tr, e_tr]; cols_te = np.c_[cols_te, lr.decision_function(feats_fit[te_i])]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1).fit(cols_tr, y[tr_i])
        oof[te_i] = rf.predict_proba(cols_te)[:, 1]
        for g in out:
            xs = np.c_[M[g], lr.decision_function(scaler.transform(F(g)))] if expert else M[g]
            out[g] += rf.predict_proba(xs)[:, 1] / folds
    res = {k_: P[k_] for k_ in P.files}
    for g, part in zip(fit_g, np.split(oof, np.cumsum(sizes)[:-1])): res[f"{g}_v4score"] = part
    for g in out: res[f"{g}_v4score"] = out[g]
    res["v4_measure_names"] = np.array(names + (["outlier_expert"] if expert else []))
    np.savez_compressed(out_path, **res)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-expert", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); n = run(a.pred, a.out, a.k, not a.no_expert, a.seed); print("measures:", n, "-> saved", a.out)

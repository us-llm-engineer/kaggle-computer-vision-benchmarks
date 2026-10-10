"""Shared pieces of the out-of-sample improvement round: a split with three separate held-back category sets, the all-image competition-style score,
the novelty scores of the four variants, bootstrap intervals and a seed band.

Why three sets: scores, checkpoints and hyper-parameters are chosen on SELECT, only V4 learns from POOL, and REPORT is read once per frozen
configuration. A novel image is any image that shows a held-out category (it may also show seen categories: 'mixed')."""
import numpy as np
from scipy.special import digamma, expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import fn_common as C

GROUPS = ("report", "select", "pool")


def make_oos_split(Y, seed=0, n_select=4, n_pool=4, count_range=(30, 150), val_fraction=0.15, report_classes=None):
    """Return a dict of index arrays: train, val_in, report, select, pool, plus the category lists of each held-back set (0-based columns).

    REPORT keeps the six categories of the first attempt (same seed), so numbers stay comparable. SELECT and POOL are drawn from the remaining
    categories whose image count lies in count_range. Every image that shows a held-out category leaves the in-distribution pool; an image showing
    categories of several sets goes to the highest-priority set (report > select > pool), so POOL images never show a REPORT or SELECT category."""
    counts = Y.sum(0)
    rep = sorted(report_classes) if report_classes is not None else C.make_split(Y, seed=seed)["novel_classes"]
    cand = [c for c in range(C.NCLS) if count_range[0] <= counts[c] <= count_range[1] and c not in rep]
    rng = np.random.RandomState(seed + 101)
    pick = rng.choice(cand, size=n_select + n_pool, replace=False).tolist()
    sel, pool = sorted(pick[:n_select]), sorted(pick[n_select:])
    touches = lambda cs: Y[:, cs].sum(1) > 0
    t_rep, t_sel, t_pool = touches(rep), touches(sel), touches(pool)
    g_rep = t_rep
    g_sel = t_sel & ~g_rep
    g_pool = t_pool & ~g_rep & ~g_sel
    held = g_rep | g_sel | g_pool
    ident = np.where(~held)[0]
    masked = np.where(Y[ident] > 0, counts[None, :], np.inf); rarest = masked.argmin(1)
    values, freq = np.unique(rarest, return_counts=True); ok = set(values[freq >= 2].tolist())
    strata = np.array([c if c in ok else -1 for c in rarest])
    train, val_in = train_test_split(ident, test_size=val_fraction, stratify=strata, random_state=seed)
    return {"train": np.sort(train), "val_in": np.sort(val_in), "report": np.where(g_rep)[0], "select": np.where(g_sel)[0], "pool": np.where(g_pool)[0],
            "report_classes": rep, "select_classes": sel, "pool_classes": pool}


def seen_columns(Y, idx):
    """Categories with at least one positive among the given images (the seen categories of a training set)."""
    return np.where(Y[idx].sum(0) > 0)[0]


def mixed_mask(Y, idx, seen):
    """For novel images: True when the image also shows a seen category."""
    return Y[idx][:, seen].sum(1) > 0


# ------------------------------------------------------------------------------------------------------------------------ metrics
def all_image_map(scores, Y):
    """MAP@20 over every given image (seen and novel together), the competition's averaging."""
    return C.map_at_k(scores, Y)


def competition_score(sauc_value, map_all):
    return 0.5 * (sauc_value + map_all)


def novelty_sauc(score, is_novel):
    return C.sauc(score, is_novel)


def group_sauc(score_in, score_group):
    """sAUC with the in-distribution validation images as negatives and one novel group as positives (higher score = more novel)."""
    s = np.r_[score_in, score_group]; y = np.r_[np.zeros(len(score_in), int), np.ones(len(score_group), int)]
    return 2.0 * roc_auc_score(y, s) - 1.0


def bootstrap_ci(fn, n_items, n_boot=500, seed=0, alpha=0.05):
    """Percentile interval of fn(index array) over bootstrap resamples of range(n_items)."""
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n_boot):
        vals.append(fn(rng.randint(0, n_items, n_items)))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def seed_band(values):
    """(mean, half-range) of a list of per-seed values; the half-range is the noise a claimed gain has to exceed."""
    v = np.asarray(values, float)
    return float(v.mean()), float((v.max() - v.min()) / 2)


# ------------------------------------------------------------------------------------------------------------------------ novelty scores
def joint_energy_novelty(logits):
    """Negative JointEnergy: -sum_i log(1 + exp(f_i)). Higher = more novel."""
    return -np.logaddexp(0.0, np.asarray(logits, np.float64)).sum(1)


def tscaled_max_novelty(logits, T=2.0):
    """1 - max_i sigmoid(f_i / T); higher = more novel."""
    return 1.0 - expit(np.asarray(logits, np.float64) / T).max(1)


def entropy_novelty(logits):
    return C.ood_scores(logits)["entropy"]


def evidence_scores(alpha, beta, lam=0.5):
    """Sum-aggregated evidential scores of the Beta evidential network; alpha, beta >= 1 are (n, L). Returns U_p^s, U_n^s and their blend (higher = more novel)."""
    alpha = np.asarray(alpha, np.float64); beta = np.asarray(beta, np.float64); L = alpha.shape[1]
    up = L / alpha.sum(1)
    un = 1.0 - (1.0 / beta).sum(1) / L
    return {"u_p": up, "u_n": un, "u_pn": lam * up + (1.0 - lam) * un}


def beta_risk(alpha, beta, y):
    """Bayes risk of the binary cross-entropy under Beta(alpha, beta): y (psi(a+b) - psi(a)) + (1-y) (psi(a+b) - psi(b)), mean over labels."""
    a = np.asarray(alpha, np.float64); b = np.asarray(beta, np.float64); y = np.asarray(y, np.float64)
    return float((y * (digamma(a + b) - digamma(a)) + (1 - y) * (digamma(a + b) - digamma(b))).mean())


def radius_novelty(feats, centre):
    """BootOOD trains pseudo-outliers to lie at a SMALLER radius around the in-distribution centre than real images, so novelty = -||h - centre||."""
    return -np.linalg.norm(np.asarray(feats, np.float64) - centre[None], axis=1)


def feature_norm_novelty(feats):
    return -np.linalg.norm(np.asarray(feats, np.float64), axis=1)


def candidate_scores(logits, feats=None, centre=None, alpha=None, beta=None):
    """Every novelty score a variant can offer, keyed by name (all: higher = more novel)."""
    out = {"msp": C.ood_scores(logits)["msp"], "joint_energy": joint_energy_novelty(logits), "tmax": tscaled_max_novelty(logits), "entropy": entropy_novelty(logits),
           "max_energy": C.ood_scores(logits)["energy"]}
    if feats is not None:
        out["feat_norm"] = feature_norm_novelty(feats)
        if centre is not None: out["radius"] = radius_novelty(feats, centre)
    if alpha is not None:
        for k, v in evidence_scores(alpha, beta, 0.5).items(): out["ev_" + k] = v
    return out


# ------------------------------------------------------------------------------------------------------------------------ geometry helpers (V1)
def banerjee_kappa(rbar, d):
    """Approximate vMF concentration from the mean resultant length."""
    rbar = np.clip(rbar, 1e-3, 0.999)
    return rbar * (d - rbar ** 2) / (1.0 - rbar ** 2)


def sample_annulus(mu, kappa, rng, d, lo=2.0, hi=3.0):
    """Virtual outlier directions at cosine t = 1 - xi / (2 kappa) from mean direction mu, with xi uniform over [d-1 + lo*s, d-1 + hi*s] and
    s = sqrt(2 (d-1)) (the 2 to 3 sigma tail of chi-square(d-1)); the tangential part is Gram-Schmidt orthogonalised random noise."""
    s = np.sqrt(2.0 * (d - 1)); xi = rng.uniform(d - 1 + lo * s, d - 1 + hi * s)
    t = float(np.clip(1.0 - xi / (2.0 * kappa), -1.0, 1.0))
    v = rng.randn(d); v -= (v @ mu) * mu; v /= np.linalg.norm(v) + 1e-12
    return t * mu + np.sqrt(max(1.0 - t * t, 0.0)) * v, t


def disjoint_pairs(Y, rng):
    """For each row choose a partner row whose label set is disjoint (or -1 when none found in a few tries); BootOOD pairs images of different classes."""
    n = len(Y); partner = np.full(n, -1)
    for i in range(n):
        for j in rng.permutation(n)[:10]:
            if j != i and (Y[i] & Y[j]).sum() == 0: partner[i] = j; break
    return partner

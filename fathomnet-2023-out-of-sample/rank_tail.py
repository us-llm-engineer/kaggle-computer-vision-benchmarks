"""Shared ranking tail of the out-of-sample round.

The competition's MAP@20 is taken over every image, and the categories that never occur in training have untrained outputs, so for a novel image
the top-20 list is filled with seen categories that are almost surely wrong. The tail keeps the seen-category order intact and adds the UNSEEN
categories of the predicted supercategory behind a novelty gate:

    score_unseen(c) = tau * g(x) * P(supercategory(c) | x) / n_unseen(supercategory(c)),    g(x) = share of in-distribution validation images with a lower novelty score

so a confidently seen image keeps its seen categories on top, and an image judged novel gets the unseen categories of its supercategory interleaved.
tau is chosen on the SELECT group only."""
import numpy as np

import fn_common as C


def novelty_gate(score, reference_scores):
    """Empirical CDF of a novelty score against in-distribution validation scores, in [0, 1]."""
    ref = np.sort(np.asarray(reference_scores, np.float64))
    return np.searchsorted(ref, np.asarray(score, np.float64), side="right") / max(len(ref), 1)


def tail_scores(p, gate, sc_prob, sc_of, seen, tau):
    """p: (n, 290) seen-category probabilities, gate: (n,), sc_prob: (n, n_super), sc_of: (290,) supercategory index of each category, seen: seen column indices."""
    n = p.shape[0]; out = np.array(p, np.float64, copy=True)
    unseen = np.ones(C.NCLS, bool); unseen[seen] = False
    out[:, unseen] = 0.0
    n_un = np.bincount(sc_of[unseen], minlength=sc_prob.shape[1]).astype(np.float64)
    per_col = np.zeros((n, C.NCLS))
    cols = np.where(unseen)[0]
    per_col[:, cols] = tau * gate[:, None] * sc_prob[:, sc_of[cols]] / np.maximum(n_un[sc_of[cols]], 1.0)[None]
    return out + per_col


def best_tau(p, gate, sc_prob, sc_of, seen, Y, grid=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Choose tau on a labelled selection set by all-image MAP@20; returns (tau, map_by_tau)."""
    res = {t: C.map_at_k(tail_scores(p, gate, sc_prob, sc_of, seen, t), Y) for t in grid}
    return max(res, key=res.get), res

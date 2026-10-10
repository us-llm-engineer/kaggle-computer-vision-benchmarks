"""Tests for the out-of-sample round: the three-set split and its leak mutants, the all-image MAP, the novelty scores, the Beta risk, the vMF annulus
sampler, the disjoint-label pairing and the ranking tail. Run: python3 tests/test_fn_oos_common.py"""
import os
import sys

import numpy as np
from scipy import integrate
from scipy.special import beta as beta_fn
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fn_common as C
import fn_oos_common as O
import rank_tail as RT

failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: failures.append(name)


# ---------------------------------------------------------------------------------------------------------------- the split
_, Y = C.load_labels()
sp = O.make_oos_split(Y, seed=0)
held = sp["report_classes"] + sp["select_classes"] + sp["pool_classes"]
names = ("train", "val_in", "report", "select", "pool")
sets = {k: set(sp[k].tolist()) for k in names}
check("the five index sets are pairwise disjoint", all(not (sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]))
check("14 distinct held-out categories", len(set(held)) == 14 == len(held))
check("no held-out category appears in training or in ID validation", Y[sp["train"]][:, held].sum() == 0 and Y[sp["val_in"]][:, held].sum() == 0)
check("every REPORT image shows a REPORT category", (Y[sp["report"]][:, sp["report_classes"]].sum(1) > 0).all())
check("every SELECT image shows a SELECT category and no REPORT category", (Y[sp["select"]][:, sp["select_classes"]].sum(1) > 0).all() and Y[sp["select"]][:, sp["report_classes"]].sum() == 0)
check("POOL images show no REPORT or SELECT category", Y[sp["pool"]][:, sp["report_classes"] + sp["select_classes"]].sum() == 0)
check("REPORT is the six categories of the first attempt", sp["report_classes"] == C.make_split(Y, seed=0)["novel_classes"])
check("the split is deterministic for a seed", all((O.make_oos_split(Y, seed=0)[k] == sp[k]).all() for k in names))
leaky = np.union1d(sp["train"], sp["select"])
check("MUTANT (SELECT images left in training) is detected", Y[leaky][:, held].sum() > 0)
leaky2 = np.union1d(sp["pool"], sp["report"])
check("MUTANT (REPORT images reaching the POOL) is detected by the POOL purity test", Y[leaky2][:, sp["report_classes"] + sp["select_classes"]].sum() > 0)
seen = O.seen_columns(Y, sp["train"])
check("seen categories exclude every held-out category", len(set(seen.tolist()) & set(held)) == 0)
mm = O.mixed_mask(Y, sp["report"], seen)
check("mixed + pure partition the REPORT images", int(mm.sum()) + int((~mm).sum()) == len(sp["report"]) and mm.any() and (~mm).any())

# ---------------------------------------------------------------------------------------------------------------- metrics
rng = np.random.RandomState(0); n = 150
Yr = (rng.rand(n, C.NCLS) < 0.012).astype(np.uint8); Yr[np.arange(n), rng.randint(0, C.NCLS, n)] = 1
S = rng.randn(n, C.NCLS) + 2.5 * Yr
ref = np.mean([C.ap_list((np.argsort(-S[i])[:20] + 1).tolist(), set((np.where(Yr[i])[0] + 1).tolist())) for i in range(n)])
check("all-image MAP equals the plain-list reference", abs(O.all_image_map(S, Yr) - ref) < 1e-12)
check("competition score is the plain average", abs(O.competition_score(0.4, 0.6) - 0.5) < 1e-12)
a = rng.randn(40); b = rng.randn(30) + 1.0
check("group_sauc equals sAUC of the concatenation", abs(O.group_sauc(a, b) - C.sauc(np.r_[a, b], np.r_[np.zeros(40, bool), np.ones(30, bool)])) < 1e-12)
m, h = O.seed_band([0.30, 0.34, 0.32])
check("seed band is mean and half-range", abs(m - 0.32) < 1e-12 and abs(h - 0.02) < 1e-12)

# ---------------------------------------------------------------------------------------------------------------- novelty scores
lg = np.array([[2.0, -1.0, 0.5], [-3.0, -3.0, -3.0]])
check("joint energy novelty = -sum log(1 + e^f)", np.allclose(O.joint_energy_novelty(lg), -np.log1p(np.exp(lg)).sum(1)))
check("an all-negative image is more novel than a confident one under joint energy", O.joint_energy_novelty(lg)[1] > O.joint_energy_novelty(lg)[0])
check("T-scaled max novelty = 1 - max sigmoid(f / T)", np.allclose(O.tscaled_max_novelty(lg, 2.0), 1 - expit(lg / 2.0).max(1)))
al = np.array([[5.0, 2.0, 2.0]]); be = np.array([[2.0, 4.0, 2.0]])
ev = O.evidence_scores(al, be, 0.3)
check("evidence scores follow the formulas", np.allclose(ev["u_p"], 3 / 9.0) and np.allclose(ev["u_n"], 1 - (1 / 2 + 1 / 4 + 1 / 2) / 3) and np.allclose(ev["u_pn"], 0.3 * ev["u_p"] + 0.7 * ev["u_n"]))
# Beta risk against numerical integration of BCE under Beta(alpha, beta)
for y, a_, b_ in [(1, 3.0, 2.0), (0, 2.5, 4.0), (1, 6.0, 1.5)]:
    pdf = lambda p: p ** (a_ - 1) * (1 - p) ** (b_ - 1) / beta_fn(a_, b_)
    bce = (lambda p: -np.log(p)) if y == 1 else (lambda p: -np.log(1 - p))
    num = integrate.quad(lambda p: bce(p) * pdf(p), 0, 1)[0]
    check(f"Beta risk matches numerical integration (y={y}, alpha={a_}, beta={b_})", abs(O.beta_risk(np.array([[a_]]), np.array([[b_]]), np.array([[y]])) - num) < 1e-6)

# ---------------------------------------------------------------------------------------------------------------- geometry helpers
d = 64; mu = np.zeros(d); mu[0] = 1.0; r2 = np.random.RandomState(1)
for kappa in (150.0, 400.0):
    z, t = O.sample_annulus(mu, kappa, r2, d)
    check(f"annulus sample is a unit vector at cosine t (kappa={kappa})", abs(np.linalg.norm(z) - 1) < 1e-9 and abs(z @ mu - t) < 1e-9)
xi_lo, xi_hi = (d - 1) + 2 * np.sqrt(2 * (d - 1)), (d - 1) + 3 * np.sqrt(2 * (d - 1))
ts = np.array([O.sample_annulus(mu, 300.0, r2, d)[1] for _ in range(200)])
check("sampled cosines come from the 2 to 3 sigma interval", ((1 - ts) * 2 * 300.0 >= xi_lo - 1e-6).all() and ((1 - ts) * 2 * 300.0 <= xi_hi + 1e-6).all())
check("Banerjee kappa increases with the resultant length", O.banerjee_kappa(0.3, 512) < O.banerjee_kappa(0.6, 512) < O.banerjee_kappa(0.9, 512))
Yp = (np.random.RandomState(2).rand(60, 30) < 0.1).astype(np.uint8)
part = O.disjoint_pairs(Yp, np.random.RandomState(3))
ok = [(Yp[i] & Yp[j]).sum() == 0 for i, j in enumerate(part) if j >= 0]
check("mixup partners never share a label", len(ok) > 0 and all(ok))
check("MUTANT (partner chosen without the label check) shares a label somewhere", any((Yp[i] & Yp[(i + 1) % 60]).sum() > 0 for i in range(60)))

# ---------------------------------------------------------------------------------------------------------------- ranking tail
sc_of, _ = C.supercat_index()
p = np.random.RandomState(4).rand(5, C.NCLS) * 0.2; sc_prob = np.random.RandomState(5).dirichlet(np.ones(20), 5)
gate = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
for tau in (0.0, 3.0):
    t = RT.tail_scores(p, gate, sc_prob, sc_of, seen, tau)
    check(f"tail leaves every seen-category score unchanged (tau={tau})", np.allclose(t[:, seen], p[:, seen]))
t = RT.tail_scores(p, gate, sc_prob, sc_of, seen, 3.0); unseen = np.setdiff1d(np.arange(C.NCLS), seen)
check("a gate of 0 adds nothing for unseen categories", np.allclose(t[0, unseen], 0.0))
c_hi = unseen[np.argmax(sc_prob[4, sc_of[unseen]] / np.maximum(np.bincount(sc_of[unseen], minlength=20)[sc_of[unseen]], 1))]
check("the unseen category with the largest prior share gets the largest tail score for a fully novel image", t[4, c_hi] == t[4, unseen].max())
ref_scores = np.array([0.1, 0.2, 0.3, 0.4])
check("novelty gate is the empirical CDF", np.allclose(RT.novelty_gate([0.05, 0.25, 0.5], ref_scores), [0.0, 0.5, 1.0]))

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)

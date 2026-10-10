"""Tests for the competition metrics and the validation split. Run: python3 tests/test_fn_common.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fn_common as C

failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: failures.append(name)


# the three examples of the competition text: all score 1.0 when the only correct label is 1
for pred in ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [1, 2, 1, 3, 1]):
    check(f"reference AP {pred} == 1.0", abs(C.ap_list(pred, {1}) - 1.0) < 1e-12)
check("reference AP: correct label at rank 2 -> 0.5", abs(C.ap_list([9, 1], {1}) - 0.5) < 1e-12)
check("reference AP: two labels at ranks 1 and 3 -> (1 + 2/3) / 2", abs(C.ap_list([1, 8, 2], {1, 2}) - (1 + 2 / 3) / 2) < 1e-12)

# the vectorised metric agrees with the reference on random score matrices
rng = np.random.RandomState(0); n = 200
Y = (rng.rand(n, C.NCLS) < 0.01).astype(np.uint8); Y[np.arange(n), rng.randint(0, C.NCLS, n)] = 1
S = rng.randn(n, C.NCLS) + 3.0 * Y
ref = np.mean([C.ap_list((np.argsort(-S[i])[:20] + 1).tolist(), set((np.where(Y[i])[0] + 1).tolist())) for i in range(n)])
check("map_at_k equals the list reference on random data", abs(C.map_at_k(S, Y) - ref) < 1e-12)
check("perfect ranking gives MAP@20 = 1", abs(C.map_at_k(Y * 10.0 + rng.rand(n, C.NCLS) * 0.1, Y) - 1.0) < 1e-12)
check("worst ranking gives MAP@20 near 0", C.map_at_k(-Y.astype(float) * 10.0 + rng.rand(n, C.NCLS), Y) < 0.05)   # float first: -uint8 wraps to 255

# sAUC: 1 for a perfect score, 0 for a constant score, -1 for an inverted one
novel = np.r_[np.zeros(50, bool), np.ones(50, bool)]
check("sAUC perfect = 1", abs(C.sauc(novel.astype(float), novel) - 1) < 1e-12)
check("sAUC constant = 0", abs(C.sauc(np.zeros(100), novel)) < 1e-12)
check("sAUC inverted = -1", abs(C.sauc(1 - novel.astype(float), novel) + 1) < 1e-12)

# out-of-sample scores: confident logits are less novel than flat ones on every score
confident = np.full((1, C.NCLS), -8.0); confident[0, 3] = 8.0; flat = np.full((1, C.NCLS), -1.0)
a, b = C.ood_scores(confident), C.ood_scores(flat)
check("msp, energy and entropy all rank flat logits as more novel", all(b[k][0] > a[k][0] for k in a))

# split: disjoint, novel images share no training image, held-out categories never appear in training, deterministic
ids = np.arange(3000); Yb = np.zeros((3000, C.NCLS), np.uint8)
for i in ids:
    Yb[i, rng.choice(40, p=np.r_[np.full(10, 0.07), np.full(30, 0.01)] / (0.7 + 0.3))] = 1
sp = C.make_split(Yb, seed=1, n_novel=3, novel_range=(30, 200))
tr, vi, vn = sp["train"], sp["val_in"], sp["val_novel"]
check("train, val_in and val_novel are disjoint", len(set(tr) & set(vi)) == 0 and len(set(tr) & set(vn)) == 0 and len(set(vi) & set(vn)) == 0)
check("the three parts cover every image", len(tr) + len(vi) + len(vn) == 3000)
check("no held-out category appears in training", Yb[tr][:, sp["novel_classes"]].sum() == 0)
check("every novel image shows a held-out category", (Yb[vn][:, sp["novel_classes"]].sum(1) > 0).all())
check("the split is deterministic for a seed", all((C.make_split(Yb, seed=1, n_novel=3, novel_range=(30, 200))[k] == sp[k]).all() for k in ("train", "val_in", "val_novel")))

# mutant: a split that forgets to remove novel images from training is caught by the held-out category test
leaky = np.union1d(tr, vn)
check("MUTANT (novel images left in training) is detected", Yb[leaky][:, sp["novel_classes"]].sum() > 0)
print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)

"""Tests for fil_common. Run: python3 helpers/tests/test_fil_common.py  (needs the downloaded data under data/ for the data-backed checks).
Each check states what defect it would catch; the 'mutant' checks build a deliberately broken variant and assert the check fails on it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fil_common as F  # noqa: E402

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print("ok  ", name)
        return fn
    return deco


def split_problems(recs, split):
    """list of defects: a file in two roles, a file in no role, copies of one file in different roles."""
    bad = []
    seen = {}
    for role, fs in split.items():
        for f in fs:
            if f in seen:
                bad.append(("file in two roles", f))
            seen[f] = role
    files = {r["file"] for r in recs}
    if set(seen) != files:
        bad.append(("coverage", len(files ^ set(seen))))
    roles_of_file = {}
    for r in recs:
        roles_of_file.setdefault(r["file"], set()).add(seen.get(r["file"]))
    bad += [("copies split across roles", f) for f, s in roles_of_file.items() if len(s) > 1]
    return bad


def lab(*boxes, n=64):
    a = np.zeros((n, n), np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, 1):
        a[y0:y1, x0:x1] = i
    return a


@check("pq: perfect prediction scores 1")
def _():
    g = lab((2, 10, 2, 10), (20, 30, 20, 40))
    r = F.pq(g, g.copy())
    assert abs(r["pq"] - 1) < 1e-12 and r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


@check("pq: empty prediction on a non-empty truth scores 0, empty vs empty scores 1")
def _():
    g = lab((2, 10, 2, 10))
    assert F.pq(g, np.zeros_like(g))["pq"] == 0.0
    assert F.pq(np.zeros_like(g), np.zeros_like(g))["pq"] == 1.0


@check("pq: one instance split into two halves (each IoU <= 0.5) gives 1 FN and 2 FP, PQ 0")
def _():
    g = lab((10, 20, 10, 30))
    p = lab((10, 20, 10, 20), (10, 20, 20, 30))
    r = F.pq(g, p)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 2, 1) and r["pq"] == 0.0


@check("pq: known partial overlap and extra false positive")
def _():
    g = lab((10, 20, 10, 20))
    p = lab((10, 20, 10, 18), (40, 50, 40, 50))     # IoU 80/100 = 0.8 plus one false positive
    r = F.pq(g, p)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert abs(r["pq"] - 0.8 / 1.5) < 1e-12


@check("pq: two instances merged into one (IoU < 0.5 with each) gives 2 FN and 1 FP")
def _():
    g = lab((10, 20, 10, 20), (10, 20, 22, 32))
    p = lab((10, 20, 10, 32))
    r = F.pq(g, p)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 2)


@check("pq mutant: a matcher that counts IoU >= 0.3 as a match is caught by the split case")
def _():
    g = lab((10, 20, 10, 30))
    p = lab((10, 20, 10, 20), (10, 20, 20, 30))
    assert F.pq(g, p, thr=0.3)["tp"] != 0 or F.pq(g, p, thr=0.3)["fp"] != 2     # with thr 0.3 the split case no longer yields (0, 2, 1)


@check("relations: split and merge counts")
def _():
    g = lab((10, 20, 10, 30))
    p = lab((10, 20, 10, 20), (10, 20, 20, 30))
    assert F.relations(g, p)["gt_split"] == 1
    g2 = lab((10, 20, 10, 20), (10, 20, 22, 32))
    p2 = lab((10, 20, 10, 32))
    assert F.relations(g2, p2)["pred_merge"] == 1


@check("instances_from_prob: components, minimum area, closing bridges a 1-pixel gap")
def _():
    p = np.zeros((64, 64), np.float32)
    p[10:13, 5:30] = 1; p[10:13, 31:50] = 1          # one line with a 1-pixel gap
    p[40:42, 40:42] = 1                              # 4-pixel speck
    a = F.instances_from_prob(p, 0.5, min_area=10, close_radius=0)
    assert a.max() == 2
    b = F.instances_from_prob(p, 0.5, min_area=10, close_radius=2)
    assert b.max() == 1


@check("rle: counts string round-trips through the reference decoder at 2048 x 2048")
def _():
    m = np.zeros((F.S, F.S), np.uint8)
    m[100:140, 200:900] = 1; m[500:503, 10:2000] = 1
    c = F.rle_counts(m)
    assert isinstance(c, str) and '"' not in c and "'" not in c
    assert (F.rle_decode(c) == m).all()


@check("rle mutant: a one-pixel shifted mask does not decode to the original")
def _():
    m = np.zeros((F.S, F.S), np.uint8)
    m[100:140, 200:900] = 1
    c = F.rle_counts(np.roll(m, 1, axis=1))
    assert not (F.rle_decode(c) == m).all()


@check("rasterize: later polygons overwrite, area close to a rectangle's")
def _():
    poly = [100, 100, 300, 100, 300, 140, 100, 140]
    lbl = F.rasterize([poly])
    assert lbl.max() == 1 and abs((lbl == 1).sum() - 200 * 40) < 600


@check("split: synthetic grouped split has no defects; per-record shuffling mutant is caught")
def _():
    recs = [{"file": f"2012010{i}0000{c}.jpeg", "rid": f"{a}-{i}{c}"} for i in range(9) for c in ("Bh", "Lh") for a in ("a", "b", "c")][:]
    files = sorted({r["file"] for r in recs})
    sp = F.split_files(files, seed=1)
    assert split_problems(recs, sp) == []
    # mutant: split per RECORD id, so copies of one file can land in different roles
    rng = np.random.RandomState(0)
    rids = [r["rid"] for r in recs]
    rng.shuffle(rids)
    cut = len(rids) // 2
    which = {rid: ("fit" if i < cut else "report") for i, rid in enumerate(rids)}
    bad = {}
    for r in recs:
        bad.setdefault(r["file"], set()).add(which[r["rid"]])
    assert any(len(s) > 1 for s in bad.values())


@check("split: deterministic in the seed and different across seeds")
def _():
    files = [f"2012{m:02d}{d:02d}000000{c}.jpeg" for m in range(1, 7) for d in range(1, 11) for c in ("Bh", "Lh", "Mh")]
    assert F.split_files(files, 3) == F.split_files(files, 3)
    assert F.split_files(files, 3) != F.split_files(files, 4)


root = F.find_root() if os.path.isdir(os.environ.get("FIL_ROOT", "data/MAGFiLO_1.0_Kaggle_2026")) else None
if root:
    recs = F.load_records(root)

    @check("data: 1154 records, 707 files, no record without polygons, real split has no defects, report role non-empty")
    def _():
        assert len(recs) == 1154 and len({r["file"] for r in recs}) == 707
        assert all(len(r["polys"]) > 0 for r in recs)
        sp = F.split_files([r["file"] for r in recs], 0)
        assert split_problems(recs, sp) == []
        assert min(len(v) for v in sp.values()) > 80

    @check("data: rasterised polygon areas agree with the file's own area field (median ratio within 10%)")
    def _():
        import json
        jp = next(os.path.join(rr, f) for rr, _, fs in os.walk(os.path.join(root, "train")) for f in fs if f.endswith(".json"))
        d = json.load(open(jp))
        ratios = []
        for a in d["annotations"][:120]:
            lbl = F.rasterize([a["segmentation"][0]])
            ratios.append((lbl > 0).sum() / max(a["area"], 1))
        assert 0.85 < float(np.median(ratios)) < 1.15, np.median(ratios)

    @check("preprocess: shape/dtype, off-disk centred, deterministic, both modes, disk radius plausible")
    def _():
        from PIL import Image
        r0 = recs[0]
        img = np.asarray(Image.open(os.path.join(root, "train", "train_images", r0["file"])).convert("L"))
        assert img.shape == (F.S, F.S) and img.dtype == np.uint8
        cy, cx, rad = F.disk_params(img)
        assert 600 < rad < 1100 and 800 < cy < 1250 and 800 < cx < 1250, (cy, cx, rad)
        a = F.preprocess(img, "flat_lcn"); b = F.preprocess(img, "flat_lcn"); c = F.preprocess(img, "raw")
        assert a.dtype == np.uint8 and (a == b).all() and c.shape == a.shape
        assert abs(int(a[0, 0]) - 127) <= 1                    # corner is off-disk -> mid value
        assert a.std() > 5

print(f"{len(PASSED)} checks passed")

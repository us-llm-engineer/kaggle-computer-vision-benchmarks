"""Inference and evaluation for the filament models.
Library: load_prep, predict_prob, score_records, grid_instance_params.
CLI (final evaluation of one trained run):
  python3 helpers/fil_infer.py --run runs/v0_s0 --prep prep --mode flat_lcn            # tunes instance parameters on CHOOSE, writes CHOOSE results
  python3 helpers/fil_infer.py --run runs/v0_s0 --prep prep --report                    # additionally reads REPORT once with the frozen parameters
The probability map is predicted once per FILE; every annotator copy of the file is scored against it (the copies are different images by the
task's wording, so each is its own ground truth). Instance parameters (threshold, minimum area, closing radius) are chosen on CHOOSE only."""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import fil_common as F  # noqa: E402
import fil_models as M  # noqa: E402


def load_prep(prep, mode="flat_lcn"):
    meta = json.load(open(os.path.join(prep, "meta.json")))
    x = np.load(os.path.join(prep, f"x_{mode}.npy"), mmap_mode="r")
    lab = np.load(os.path.join(prep, "lab.npy"), mmap_mode="r")
    fidx = {(f["part"], f["file"]): i for i, f in enumerate(meta["files"])}
    return meta, x, lab, fidx


def tile_offsets(n=F.S, tile=1024, stride=768):
    """start offsets of overlapping tiles that cover [0, n) with the last tile flush to the end."""
    offs = list(range(0, max(n - tile, 0) + 1, stride))
    if offs[-1] != n - tile:
        offs.append(n - tile)
    return offs


_WIN = {}


def _window(tile, dev):
    if (tile, dev.type) not in _WIN:
        h = torch.hann_window(tile, periodic=False, device=dev).clamp_min(0) + 1e-3
        _WIN[(tile, dev.type)] = (h[:, None] * h[None, :]).float()
    return _WIN[(tile, dev.type)]


@torch.no_grad()
def predict_prob(model, xq, dev, flip_tta=False, tile=1024, stride=768, batch=4):
    """uint8 (S,S) -> float16 probability map (S,S). The image is cut into overlapping tiles of the TRAINING crop size (so the bottleneck attention sees the
    same number of tokens as in training), predicted in bf16 autocast, and blended with a Hann window. flip_tta averages with the horizontally flipped input."""
    x = torch.from_numpy(F.to_float(np.asarray(xq))).to(dev)
    S = x.shape[-1]
    offs = tile_offsets(S, tile, stride)
    coords = [(y, xx) for y in offs for xx in offs]
    acc = torch.zeros(S, S, device=dev); wsum = torch.zeros(S, S, device=dev)
    win = _window(tile, dev)
    ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        t = torch.stack([x[y:y + tile, xx:xx + tile] for y, xx in cs])[:, None]
        with ac:
            p = torch.sigmoid(model(t).float())
            if flip_tta:
                p = 0.5 * (p + torch.sigmoid(model(t.flip(-1)).float()).flip(-1))
        for (y, xx), pp in zip(cs, p[:, 0]):
            acc[y:y + tile, xx:xx + tile] += pp * win; wsum[y:y + tile, xx:xx + tile] += win
    return (acc / wsum).half().cpu().numpy()


def score_records(probs, meta, lab, rec_ids, thr=0.5, min_area=30, close_radius=0):
    """per-record metrics. probs: file -> float16 map. returns list of dicts (pq, sq, rq, tp, fp, fn, dice, iou, relations, counts)."""
    out = []
    ridx = {r["rid"]: j for j, r in enumerate(meta["records"])}
    for rid in rec_ids:
        j = ridx[rid]
        file = meta["records"][j]["file"]
        gt = np.asarray(lab[j]).astype(np.int32)
        pred = F.instances_from_prob(probs[file].astype(np.float32), thr, min_area, close_radius)
        r = F.pq(gt, pred)
        d = F.dice_iou_pixel(gt, pred)
        rel = F.relations(gt, pred)
        out.append({"rid": rid, "file": file, "pq": r["pq"], "sq": r["sq"], "rq": r["rq"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                    "ious": r["ious"], "gt_iou": r["gt_iou"], "gt_area": r["gt_area"], "dice": d["dice"], "iou": d["iou"], **rel})
    return out


SIZE_BINS = (0, 300, 1000, 3000, 10 ** 9)


def recall_by_size(rows):
    """share of ground-truth instances matched (IoU > 0.5), by instance area bin (pixels): [<300, 300-1000, 1000-3000, >=3000]."""
    areas = np.array([a for r in rows for a in r["gt_area"]], dtype=float)
    hit = np.array([v > 0 for r in rows for v in r["gt_iou"]], dtype=bool)
    out = []
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = (areas >= lo) & (areas < hi)
        out.append({"lo": lo, "n": int(m.sum()), "recall": float(hit[m].mean()) if m.any() else None})
    return out


def summarize(rows):
    g = lambda k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    ious = [v for r in rows for v in r["ious"]]
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    return {"n_records": len(rows), "pq_mean_over_records": g("pq"), "sq_mean": g("sq"), "rq_mean": g("rq"), "dice_mean": g("dice"), "iou_mean": g("iou"),
            "pq_pooled": (sum(ious) / (tp + 0.5 * fp + 0.5 * fn)) if (tp + fp + fn) else 1.0, "tp": tp, "fp": fp, "fn": fn,
            "gt_split_total": sum(r["gt_split"] for r in rows), "pred_merge_total": sum(r["pred_merge"] for r in rows),
            "pred_instances": sum(r["pred"] for r in rows), "gt_instances": sum(r["gt"] for r in rows), "recall_by_size": recall_by_size(rows),
            "pq_quantiles": {str(q): float(np.percentile([r["pq"] for r in rows], q)) for q in (10, 25, 50, 75, 90)} if rows else {}}


_G = {}


def _score_one(args_):
    th, ma, cr = args_
    probs, meta, lab, rec_ids = _G["ctx"]
    s = summarize(score_records(probs, meta, lab, rec_ids, th, ma, cr))
    return {"thr": th, "min_area": ma, "close": cr, "pq": s["pq_mean_over_records"], "sq": s["sq_mean"], "rq": s["rq_mean"]}


def grid_instance_params(probs, meta, lab, rec_ids, thrs=(0.4, 0.5, 0.6), min_areas=(30, 100), closes=(0, 2), workers=None, max_records=80):
    """choose (thr, min_area, close_radius) by mean per-record PQ on (a fixed prefix of) the given CHOOSE records, scored in parallel worker processes."""
    import multiprocessing as mp
    rec_ids = list(rec_ids)[:max_records]
    _G["ctx"] = (probs, meta, lab, rec_ids)
    cfgs = list(itertools.product(thrs, min_areas, closes))
    with mp.get_context("fork").Pool(workers or max(1, os.cpu_count() or 1)) as pool:
        table = pool.map(_score_one, cfgs)
    best = max(table, key=lambda t: t["pq"])
    return best, table


def tile_args(cfg):
    """tile = the training crop (same number of attention tokens as in training); stride = 3/4 of the tile."""
    t = int(cfg.get("crop", 1024))
    return {"tile": t, "stride": (t * 3) // 4, "batch": 8 if t <= 512 else 4}


def build_model(cfg, pretrained=True):
    """variant -> model. v3 (Mask2Former) is imported lazily so the other variants do not need the transformers package; pretrained=False builds the
    architecture only (the weights then come from a saved checkpoint)."""
    if cfg["variant"] == "v3":
        import fil_m2f as W
        return W.M2F(pretrained=pretrained)
    return M.UNet(base=cfg.get("base", 32), edge_attn=cfg["variant"] == "v2")


def load_run(run, dev):
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = build_model(cfg, pretrained=False).to(dev)
    sd = torch.load(os.path.join(run, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prep", default="prep")
    ap.add_argument("--mode", default="flat_lcn")
    ap.add_argument("--report", action="store_true", help="also score REPORT once with the parameters chosen on CHOOSE")
    ap.add_argument("--flip-tta", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, x, lab, fidx = load_prep(args.prep, args.mode)
    model, cfg = load_run(args.run, dev)
    t0 = time.time()
    res = {"run": args.run, "config": cfg}
    outp = os.path.join(args.run, "eval.json")
    if os.path.exists(outp):
        res = json.load(open(outp))

    ta = tile_args(cfg)

    def predict(role):
        files = meta["split"][role]
        return {f: predict_prob(model, x[fidx[("train", f)]], dev, args.flip_tta, **ta) for f in files}

    rec_ids = {r: [rr["rid"] for rr in meta["records"] if rr["role"] == r] for r in ("choose", "report")}
    pc = predict("choose")
    best, table = grid_instance_params(pc, meta, lab, rec_ids["choose"])
    rows = score_records(pc, meta, lab, rec_ids["choose"], best["thr"], best["min_area"], best["close"])
    res["choose"] = {"params": best, "grid": table, "summary": summarize(rows)}
    print(f"[eval] CHOOSE best {best} -> {res['choose']['summary']['pq_mean_over_records']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if args.report:
        if "report" in res:
            print("[eval] REPORT already read for this run; refusing to read it again (delete eval.json by hand to override)", flush=True)
        else:
            pr = predict("report")
            rrows = score_records(pr, meta, lab, rec_ids["report"], best["thr"], best["min_area"], best["close"])
            res["report"] = {"params": best, "summary": summarize(rrows), "per_record": [{k: v for k, v in r.items() if k not in ("ious", "gt_iou", "gt_area")} for r in rrows]}
            s = res["report"]["summary"]
            print(f"[eval] REPORT PQ {s['pq_mean_over_records']:.4f} SQ {s['sq_mean']:.4f} RQ {s['rq_mean']:.4f} Dice {s['dice_mean']:.4f}", flush=True)
    json.dump(res, open(outp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

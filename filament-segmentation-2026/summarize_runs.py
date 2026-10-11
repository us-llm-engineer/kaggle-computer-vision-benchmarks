"""Write the two small result tables from the saved per-run statistics.
usage: python3 summarize_runs.py [--runs results/runs] [--out results]
  run_summary.csv          one row per run: epochs, best epoch, best smoothed CHOOSE PQ and the metrics of the best checkpoint
  first_pass_30_epochs.csv seed-0 runs read at epoch 30, the end of the first cosine cycle (the whole run for the 9-epoch Mask2Former job)"""
import argparse
import csv
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--out", default="results")
a = ap.parse_args()
cols = ["run", "variant", "seed", "epochs_trained", "best_epoch", "best_smoothed_choose_pq", "choose_pq", "choose_sq", "choose_rq", "choose_dice", "pred_instances", "gt_instances", "tp", "fp", "fn",
        "gt_split", "pred_merge", "monitoring_records", "parameters"]


def row(name, epochs, ev, params, cfg, ids):
    b = max(ev, key=lambda e: e["smoothed_pq"]); c = b["choose"]
    return {"run": name, "variant": name.split("_")[0], "seed": cfg["seed"], "epochs_trained": epochs, "best_epoch": b["epoch"], "best_smoothed_choose_pq": round(b["smoothed_pq"], 4),
            "choose_pq": round(c["pq_mean_over_records"], 4), "choose_sq": round(c["sq_mean"], 4), "choose_rq": round(c["rq_mean"], 4), "choose_dice": round(c["dice_mean"], 4),
            "pred_instances": c["pred_instances"], "gt_instances": c["gt_instances"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "gt_split": c["gt_split_total"],
            "pred_merge": c["pred_merge_total"], "monitoring_records": len(ids), "parameters": params}


full, first = [], []
for d in sorted(glob.glob(os.path.join(a.runs, "v*_s*"))):
    name = os.path.basename(d); H = json.load(open(os.path.join(d, "history.json"))); cfg = json.load(open(os.path.join(d, "config.json")))
    ids = json.load(open(os.path.join(d, "best_extra.json")))["eval_ids"]
    ep = H["epochs"]; ev = [e for e in ep if "smoothed_pq" in e]
    full.append(row(name, ep[-1]["epoch"], ev, H["params"], cfg, ids))
    if cfg["seed"] == 0:                              # only seed 0 had the 30-epoch first cycle; seeds 1 and 2 were single 60-epoch cycles
        ev30 = [e for e in ev if e["epoch"] <= 30]
        first.append(row(name, min(30, ep[-1]["epoch"]), ev30, H["params"], cfg, ids))
for fn, rows in (("run_summary.csv", full), ("first_pass_30_epochs.csv", first)):
    with open(os.path.join(a.out, fn), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(a.out, fn), len(rows), "rows")

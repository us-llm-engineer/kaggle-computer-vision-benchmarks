"""Read-only analysis of the saved training statistics (history.json, best_extra.json): what limits PQ?
usage: python3 helpers/analyze_round1.py stats_round1   (prints compact tables; writes nothing)"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "stats_round1"
V = [v for v in ("v0", "v1", "v2", "v3") if os.path.isdir(os.path.join(root, f"{v}_s0"))]
H = {v: json.load(open(f"{root}/{v}_s0/history.json"))["epochs"] for v in V}
ev = {v: [e for e in H[v] if "choose" in e] for v in V}
print("== epochs trained / evaluations:", {v: (H[v][-1]["epoch"], len(ev[v])) for v in V})
print("\n== CHOOSE trajectory (every evaluation): epoch pq sq rq dice tp fp fn split merge pred/gt")
for v in V:
    print(v)
    for e in ev[v][::max(1, len(ev[v]) // 8)] + ev[v][-1:]:
        c = e["choose"]; print(f"  ep{e['epoch']:3d} pq {c['pq_mean_over_records']:.3f} sq {c['sq_mean']:.3f} rq {c['rq_mean']:.3f} dice {c['dice_mean']:.3f} tp {c['tp']:4d} fp {c['fp']:4d} fn {c['fn']:4d} split {c['gt_split_total']:3d} merge {c['pred_merge_total']:3d} pred/gt {c['pred_instances']}/{c['gt_instances']}")
print("\n== train vs CHOOSE Dice gap, loss slope over the last 10 epochs, lr")
for v in V:
    h = H[v]; last = h[-1]; g = last.get("train_minus_choose_dice"); l10 = [e["loss"] for e in h[-10:]]
    print(f"{v}: epochs {last['epoch']} train_dice {last['train_dice']:.3f} gap {g if g is None else round(g,3)} loss first-to-last of last 10 {l10[0]:.3f}->{l10[-1]:.3f} lr {last['lr']:.1e}")
print("\n== recall by instance size (last evaluation with the table): bins <300, 300-1000, 1000-3000, >=3000")
for v in V:
    c = ev[v][-1]["choose"]; print(v, [(b["n"], None if b["recall"] is None else round(b["recall"], 2)) for b in c["recall_by_size"]])
print("\n== PQ vs threshold (last evaluation having it) and score separation")
for v in V:
    c = next((e["choose"] for e in reversed(ev[v]) if e.get("has_extra_thr", True) and len(e["choose"]["pq_by_thr"]) > 1), ev[v][-1]["choose"])
    print(v, {k: round(x, 3) for k, x in c["pq_by_thr"].items()}, "p_in", round(c["p_in_gt"], 3), "p_out", round(c["p_out_gt"], 4), "p99_out", round(c["p99_out_gt"], 3))
print("\n== gradient norm / update ratio by group (mean over the last 5 epochs)")
for v in V:
    gn, ur = {}, {}
    for e in H[v][-5:]:
        for k, x in e["grad_norm"].items(): gn.setdefault(k, []).append(x)
        for k, x in e["update_ratio"].items(): ur.setdefault(k, []).append(x)
    print(v, {k: f"{np.mean(x):.2e}" for k, x in gn.items()}, "| upd", {k: f"{np.mean(x):.1e}" for k, x in ur.items()})
print("\n== per-record PQ at the best checkpoint (monitoring subset)")
for v in V:
    p = os.path.join(root, f"{v}_s0", "best_extra.json")
    if os.path.exists(p):
        q = np.array(json.load(open(p))["per_record_pq"]); print(f"{v}: n {len(q)} mean {q.mean():.3f} median {np.median(q):.3f} share==0 {np.mean(q == 0):.2f} share>0.5 {np.mean(q > 0.5):.2f} min {q.min():.2f} max {q.max():.2f}")
print("\n== throughput and time per epoch (median), peak memory")
for v in V:
    s = [e["seconds"] for e in H[v]]; print(v, f"median {np.median(s):.0f}s/epoch total {sum(s)/60:.1f} min, peak mem {max(e['peak_mem_gib'] for e in H[v]):.1f} GiB")

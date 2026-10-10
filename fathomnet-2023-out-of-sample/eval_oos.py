"""Out-of-sample round: evaluation of the trained variants. This is the only place REPORT images are scored.

For each variant the novelty score is chosen on SELECT (highest mean sAUC over seeds), the ranking-tail strength tau is chosen on SELECT (all-image
MAP@20), and then REPORT is read once: sAUC (all novel, pure novel, mixed), MAP@20 on seen / novel / all images, the competition-style score
0.5 * (sAUC + all-image MAP@20) with and without the tail, per seed, for the seed ensemble and with bootstrap intervals.
usage: python3 eval_oos.py --variants bce vmf radius evid --seeds 0 1 2 [--out results/oos_results.json]"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_curve

import fn_common as C
import fn_oos_common as O
import rank_tail as RT

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=["bce", "vmf", "radius", "evid"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--out", default=None); ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()
Y = np.load(f"{C.PREP}/train_y.npy"); sc_of, sc_names = C.supercat_index()
LAMS = (0.1, 0.3, 0.5, 0.7, 0.9)
# Novelty score of each variant, fixed before the rerun: V0 the competition baseline 1 - max probability; V1 energy (VMF-GOS); V2 distance to the ID centre (BootOOD feature
# geometry); V3 the evidential blend at the paper's default lambda_2 = 0.5; V4 the combiner. All other scores are reported descriptively (report_sauc_by_score), not selected.
FIXED_SCORE = {"bce": "msp", "vmf": "joint_energy", "radius": "radius", "evid": "ev_u_pn", "comb": "v4_combiner"}


def scores_of(P, g, variant):
    """All novelty scores of one prediction file for one group (higher = more novel)."""
    lg = P[f"{g}_logit"].astype(np.float64); ft = P[f"{g}_feat"].astype(np.float64)
    cen = P["centre"] if variant == "radius" else None
    al = P[f"{g}_alpha"].astype(np.float64) if f"{g}_alpha" in P.files else None; be = P[f"{g}_beta"].astype(np.float64) if f"{g}_beta" in P.files else None
    s = O.candidate_scores(lg, ft, cen, al, be)
    if al is not None:
        for lam in LAMS: s[f"ev_u_pn@{lam}"] = O.evidence_scores(al, be, lam)["u_pn"]
    if f"{g}_v4score" in P.files: s["v4_combiner"] = P[f"{g}_v4score"].astype(np.float64)
    return s


def evaluate(logit, sc, scores_in, scores_nov, idx_in, idx_nov, sel_pack, seen, key):
    """Metrics of one frozen configuration. scores_*: dict key -> array; sel_pack carries what the tail needs from SELECT."""
    s_in, s_nov = scores_in[key], scores_nov[key]
    sa = O.group_sauc(s_in, s_nov)
    mixed = O.mixed_mask(Y, idx_nov, seen)
    sa_pure = O.group_sauc(s_in, s_nov[~mixed]) if (~mixed).any() else None; sa_mixed = O.group_sauc(s_in, s_nov[mixed]) if mixed.any() else None
    p_in, p_nov = logit["in"], logit["nov"]
    Yin, Ynov = Y[idx_in], Y[idx_nov]
    map_seen = C.map_at_k(p_in, Yin); map_nov = C.map_at_k(p_nov, Ynov); map_all = C.map_at_k(np.r_[p_in, p_nov], np.r_[Yin, Ynov])
    out = {"sauc_all": sa, "sauc_pure": sa_pure, "sauc_mixed": sa_mixed, "map_seen": map_seen, "map_novel": map_nov, "map_all": map_all, "score": O.competition_score(sa, map_all)}
    tau = sel_pack["tau"]
    gate_in = RT.novelty_gate(s_in, s_in); gate_nov = RT.novelty_gate(s_nov, s_in)
    sp_in = sc["in"]; sp_nov = sc["nov"]
    t_in = RT.tail_scores(p_in, gate_in, sp_in, sc_of, seen, tau); t_nov = RT.tail_scores(p_nov, gate_nov, sp_nov, sc_of, seen, tau)
    map_all_tail = C.map_at_k(np.r_[t_in, t_nov], np.r_[Yin, Ynov])
    out.update({"tau": tau, "map_novel_tail": C.map_at_k(t_nov, Ynov), "map_all_tail": map_all_tail, "score_tail": O.competition_score(sa, map_all_tail)})
    return out


def sc_prior(P, g):
    p = 1 / (1 + np.exp(-P[f"{g}_sc"].astype(np.float64))); return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


results = {}
for v in args.variants:
    preds = [np.load(f"{C.OUT}/{v}_s{s}_pred.npz") for s in args.seeds]
    P0 = preds[0]; seen = P0["seen"]; idx = {g: P0[f"{g}_idx"] for g in ("val_in", "select", "report")}
    per_seed = [{g: scores_of(P, g, v) for g in ("val_in", "select", "report")} for P in preds]
    keys = list(per_seed[0]["val_in"])
    sel_mean = {k: float(np.mean([O.group_sauc(ps["val_in"][k], ps["select"][k]) for ps in per_seed])) for k in keys}
    key = FIXED_SCORE[v]       # fixed in advance from each paper's design; never searched on SELECT or REPORT
    res = {"chosen_score": key, "select_sauc_by_score": sel_mean, "per_seed": [], "seeds": args.seeds}
    soft = lambda P, g: 1 / (1 + np.exp(-P[f"{g}_logit"].astype(np.float64)))
    for P, ps in zip(preds, per_seed):
        p = {g: soft(P, g) for g in ("val_in", "select", "report")}
        # tau on SELECT by all-image MAP over val_in + select
        gate_in = RT.novelty_gate(ps["val_in"][key], ps["val_in"][key]); gate_sel = RT.novelty_gate(ps["select"][key], ps["val_in"][key])
        tau, _ = RT.best_tau(np.r_[p["val_in"], p["select"]], np.r_[gate_in, gate_sel], np.r_[sc_prior(P, "val_in"), sc_prior(P, "select")], sc_of, seen,
                             np.r_[Y[idx["val_in"]], Y[idx["select"]]])
        res["per_seed"].append(evaluate({"in": p["val_in"], "nov": p["report"]}, {"in": sc_prior(P, "val_in"), "nov": sc_prior(P, "report")}, ps["val_in"], ps["report"],
                                        idx["val_in"], idx["report"], {"tau": tau}, seen, key))
        res["per_seed"][-1]["sauc_select"] = O.group_sauc(ps["val_in"][key], ps["select"][key])
    # seed ensemble: average probabilities and per-seed scores
    p_e = {g: np.mean([soft(P, g) for P in preds], 0) for g in ("val_in", "select", "report")}
    sc_e = {g: np.mean([sc_prior(P, g) for P in preds], 0) for g in ("val_in", "select", "report")}
    s_e = {g: {k: np.mean([ps[g][k] for ps in per_seed], 0) for k in keys} for g in ("val_in", "select", "report")}
    gate_in = RT.novelty_gate(s_e["val_in"][key], s_e["val_in"][key]); gate_sel = RT.novelty_gate(s_e["select"][key], s_e["val_in"][key])
    tau, tau_curve = RT.best_tau(np.r_[p_e["val_in"], p_e["select"]], np.r_[gate_in, gate_sel], np.r_[sc_e["val_in"], sc_e["select"]], sc_of, seen, np.r_[Y[idx["val_in"]], Y[idx["select"]]])
    res["ensemble"] = evaluate({"in": p_e["val_in"], "nov": p_e["report"]}, {"in": sc_e["val_in"], "nov": sc_e["report"]}, s_e["val_in"], s_e["report"], idx["val_in"], idx["report"], {"tau": tau}, seen, key)
    res["report_sauc_by_score"] = {k: O.group_sauc(s_e["val_in"][k], s_e["report"][k]) for k in keys}
    res["tau_curve_select"] = {str(k): float(x) for k, x in tau_curve.items()}
    res["ensemble"]["sauc_select"] = O.group_sauc(s_e["val_in"][key], s_e["select"][key])
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(idx["val_in"])), np.ones(len(idx["report"]))], np.r_[s_e["val_in"][key], s_e["report"][key]])
    grid = np.linspace(0, 1, 101); res["roc"] = {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()}
    cats = []
    for c in P0["report_classes"] if "report_classes" in P0.files else []:
        m = Y[idx["report"]][:, c] > 0
        if m.sum() >= 5: cats.append({"category": int(c + 1), "supercat": sc_names[sc_of[c]], "n": int(m.sum()), "sauc": O.group_sauc(s_e["val_in"][key], s_e["report"][key][m])})
    res["per_category"] = cats
    n_in, n_nov = len(idx["val_in"]), len(idx["report"])
    Yin, Ynov = Y[idx["val_in"]], Y[idx["report"]]
    t_in = RT.tail_scores(p_e["val_in"], RT.novelty_gate(s_e["val_in"][key], s_e["val_in"][key]), sc_e["val_in"], sc_of, seen, tau)
    t_nov = RT.tail_scores(p_e["report"], RT.novelty_gate(s_e["report"][key], s_e["val_in"][key]), sc_e["report"], sc_of, seen, tau)

    def score_boot(sample, tail=True):
        rin = np.random.RandomState(int(sample[0])).randint(0, n_in, n_in); rnov = sample % n_nov
        sa = O.group_sauc(s_e["val_in"][key][rin], s_e["report"][key][rnov])
        a = np.r_[t_in[rin], t_nov[rnov]] if tail else np.r_[p_e["val_in"][rin], p_e["report"][rnov]]
        return O.competition_score(sa, C.map_at_k(a, np.r_[Yin[rin], Ynov[rnov]]))
    res["ensemble"]["score_ci95"] = O.bootstrap_ci(lambda s: score_boot(s, False), n_nov, args.boot)
    res["ensemble"]["score_tail_ci95"] = O.bootstrap_ci(lambda s: score_boot(s, True), n_nov, args.boot)
    for k in ("score", "score_tail", "sauc_all", "map_all"):
        m, h = O.seed_band([r[k] for r in res["per_seed"]]); res[f"band_{k}"] = {"mean": m, "half_range": h}
    results[v] = res
    e = res["ensemble"]
    print(f"{v:>7}: score {res['band_score']['mean']:.4f} +/- {res['band_score']['half_range']:.4f} (seeds) | ensemble {e['score']:.4f} CI {e['score_ci95'][0]:.3f}-{e['score_ci95'][1]:.3f} "
          f"| with tail {e['score_tail']:.4f} | sAUC {e['sauc_all']:.4f} (pure {e['sauc_pure']}, mixed {e['sauc_mixed']}) | MAP seen {e['map_seen']:.4f} novel {e['map_novel']:.4f} "
          f"all {e['map_all']:.4f} | score key {key} | tau {e['tau']}")
print("floor 0.55: ensemble scores", {v: round(results[v]["ensemble"]["score_tail"], 4) for v in results})
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); json.dump(results, open(args.out, "w"), indent=1, default=float)

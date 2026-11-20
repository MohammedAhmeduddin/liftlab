"""
Benchmark every LiftLab estimator on the real Criteo uplift data.

Criteo is documented as a randomized trial, but at 14M rows a classifier
can predict treatment slightly better than chance, and the features that
differ between arms also predict the outcome. So the raw difference in
means is reported but NOT used as ground truth; the reference is
cross-fitted AIPW on all rows, which adjusts for the features.

  0. Randomization check: balance and a classifier two-sample test
  1. Effects on all rows: raw difference in means and AIPW (reference)
  2. Effect by f8 value group, raw and AIPW (no extra model)
  3. Difference in means, AIPW and PSM on a model sample
  4. Confounded-sample test: bias a sample on purpose; can AIPW / PSM
     recover the reference where the naive comparison fails? (ATT, since
     treated units are untouched)
  5. CATE models (T, X, Causal Forest) fit on a train split and judged on a
     held-out split (Qini, observed uplift in the top 10% / 25%)
  6. Refutation: placebo treatment, random common cause, subset stability

Writes results/criteo_benchmark.json after every section.

    python scripts/benchmark_criteo.py --data data/criteo_uplift.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liftlab.causal.diagnostics import balance, classifier_two_sample_test  # noqa: E402
from liftlab.causal.estimators import (  # noqa: E402
    Z95, PropensityScoreMatching, aipw, aipw_scores, difference_in_means,
)
from liftlab.causal.evaluation import observed_uplift_by_group, value_bins  # noqa: E402
from liftlab.causal.refutation import (  # noqa: E402
    placebo_treatment, random_common_cause, subset_stability,
)
from liftlab.causal.simulate import make_confounded  # noqa: E402
from liftlab.causal.uplift import ESTIMATORS  # noqa: E402
from liftlab.data.loaders import FEATURES, load_criteo  # noqa: E402


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def est_dict(e) -> dict:
    return {"method": e.method, "estimand": e.estimand, "estimate": e.ate,
            "ci": [e.ci_lower, e.ci_upper], "std_error": e.std_error, "p_value": e.p_value,
            "n_treated": e.n_treated, "n_control": e.n_control, **e.metadata}


class Report:
    def __init__(self, out: Path, args):
        self.out = out
        self.data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "python": platform.python_version(), "args": vars(args), "sections": {}}

    def add(self, name: str, value, seconds: float):
        self.data["sections"][name] = {"seconds": round(seconds, 1), **to_jsonable(value)}
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.data, indent=2))
        print(f"  ✓ {name} ({seconds:.1f}s) → saved")


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    return time.perf_counter()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/criteo_uplift.csv")
    ap.add_argument("--out", default="results/criteo_benchmark.json")
    ap.add_argument("--model-frac", type=float, default=0.02,
                    help="sample used for AIPW/PSM/CATE models (default 2%% ≈ 280K rows)")
    ap.add_argument("--outcomes", nargs="+", default=["conversion", "visit"])
    ap.add_argument("--cf-trees", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    rep = Report(Path(args.out), args)
    start = time.perf_counter()

    t0 = section("Loading data")
    full = load_criteo(path=args.data)
    rep.add("data", {"rows": len(full), "treated_share": float(full.treatment.mean()),
                     "conversion_rate": float(full.conversion.mean()),
                     "visit_rate": float(full.visit.mean())}, time.perf_counter() - t0)

    t0 = section("0. Randomization check")
    bal = balance(full, FEATURES)
    c2 = classifier_two_sample_test(full, FEATURES, seed=args.seed)
    print(f"  max |SMD| {bal['max_smd']:.4f} ({bal['max_smd'] / bal['smd_noise_sd']:.0f}x the noise level)")
    print(f"  classifier AUC {c2['auc']:.4f} (shuffled labels reach {c2['null_auc_max']:.4f})"
          f" → imbalance {'DETECTED' if c2['detected'] else 'not detected'}")
    rep.add("randomization_check", {"balance": bal, "c2st": c2}, time.perf_counter() - t0)

    t0 = section("1. Effects on all rows (reference = AIPW)")
    effects, scores = {}, {}
    for y in args.outcomes:
        dm = est_dict(difference_in_means(full, outcome_col=y))
        psi, _ = aipw_scores(full, FEATURES, outcome_col=y, seed=args.seed)
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        ate = float(psi.mean())
        ref = {"method": "AIPW (doubly robust)", "estimand": "ATE", "estimate": ate,
               "ci": [ate - Z95 * se, ate + Z95 * se], "std_error": se,
               "relative_lift": ate / dm["control_rate"] if dm["control_rate"] else None}
        effects[y] = {"difference_in_means": dm, "aipw": ref,
                      "dm_minus_aipw": dm["estimate"] - ate, "reference": "aipw"}
        scores[y] = psi
        print(f"  {y:10s} raw difference {dm['estimate']:.6f} ({dm['relative_lift']:+.1%})   "
              f"AIPW {ate:.6f} CI [{ref['ci'][0]:.6f}, {ref['ci'][1]:.6f}] ({ref['relative_lift']:+.1%})")
    rep.add("full_data_effects", effects, time.perf_counter() - t0)
    target = {y: effects[y]["aipw"]["estimate"] for y in args.outcomes}

    t0 = section("2. Effect by f8 value group (all rows)")
    # Bins on VALUES: 52% of users share one f8 value, and the file is stored
    # ordered by treatment, so rank-based quartiles would split ties by file
    # position and put treated and control users in different groups.
    f8_bins, f8_edges = value_bins(full["f8"].to_numpy(), 4)
    seg = {"edges": f8_edges, "n_distinct_values": int(full["f8"].nunique()),
           "top_value_share": float(full["f8"].value_counts(normalize=True).iloc[0])}
    for y in args.outcomes:
        rows = observed_uplift_by_group(full[y].to_numpy(float), full.treatment.to_numpy(int), f8_bins)
        rows["value_from"], rows["value_to"] = f8_edges[:-1][: len(rows)], f8_edges[1:][: len(rows)]
        psi = scores[y]
        aipw_rows = []
        for g in rows["group"]:
            m = f8_bins == g
            est = float(psi[m].mean())
            se = float(psi[m].std(ddof=1) / np.sqrt(m.sum()))
            aipw_rows.append({"aipw": est, "aipw_ci_lower": est - Z95 * se, "aipw_ci_upper": est + Z95 * se,
                              "aipw_lift_vs_overall": est / target[y] if target[y] else None})
        rows = rows.join(pd.DataFrame(aipw_rows))
        seg[y] = rows.to_dict(orient="records")
        for r in seg[y]:
            print(f"  {y:10s} f8 in [{r['value_from']:.3f}, {r['value_to']:.3f}] n={r['n']:,}  "
                  f"raw {r['lift_vs_overall']:.2f}x  AIPW {r['aipw']:.6f} "
                  f"[{r['aipw_ci_lower']:.6f}, {r['aipw_ci_upper']:.6f}] = {r['aipw_lift_vs_overall']:.2f}x")
    print(f"  ({seg['n_distinct_values']:,} distinct f8 values; most common covers "
          f"{seg['top_value_share']:.1%} of users)")
    rep.add("f8_groups", seg, time.perf_counter() - t0)
    del scores

    sample = full.sample(frac=args.model_frac, random_state=args.seed).reset_index(drop=True)
    del full
    print(f"\nModel sample: {len(sample):,} rows ({args.model_frac:.0%})")

    t0 = section("3. Estimators on the model sample (vs the all-rows AIPW reference)")
    p_treat = float(sample.treatment.mean())
    adj = {}
    for y in args.outcomes:
        adj[y] = {
            "difference_in_means": est_dict(difference_in_means(sample, outcome_col=y)),
            "aipw": est_dict(aipw(sample, FEATURES, outcome_col=y, propensity=p_treat)),
            "psm": est_dict(PropensityScoreMatching(seed=args.seed)
                            .estimate(sample, FEATURES, outcome_col=y)),
        }
        for k, r in adj[y].items():
            r["covers_reference"] = bool(r["ci"][0] <= target[y] <= r["ci"][1])
            print(f"  {y:10s} {k:20s} {r['estimand']} {r['estimate']:.6f} "
                  f"CI [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]  covers reference: {r['covers_reference']}")
        adj[y]["reference"] = target[y]
    rep.add("sample_estimators", adj, time.perf_counter() - t0)

    t0 = section("4. Confounded-sample test")
    y0 = args.outcomes[0]
    controls = sample[sample.treatment == 0]
    corr = {f: abs(float(np.corrcoef(controls[f], controls[y0])[0, 1])) for f in FEATURES}
    driver = max(corr, key=corr.get)
    conf = make_confounded(sample, feature=driver, strength=1.5, seed=args.seed)
    naive = difference_in_means(conf, outcome_col=y0)
    conf_res = {
        "outcome": y0, "confounding_feature": driver, "feature_outcome_corr": corr[driver],
        "rows": len(conf), "control_share": float(1 - conf.treatment.mean()),
        "control_share_before": float(1 - sample.treatment.mean()),
        "target": target[y0],
        "naive_difference_in_means": est_dict(naive),
        "aipw_att": est_dict(aipw(conf, FEATURES, outcome_col=y0, estimand="ATT")),
        "psm": est_dict(PropensityScoreMatching(seed=args.seed)
                        .estimate(conf, FEATURES, outcome_col=y0)),
    }
    for k in ("naive_difference_in_means", "aipw_att", "psm"):
        r = conf_res[k]
        r["covers_target"] = bool(r["ci"][0] <= conf_res["target"] <= r["ci"][1])
        print(f"  {k:26s} {r['estimate']:.6f}  CI [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]  "
              f"covers target {conf_res['target']:.6f}: {r['covers_target']}")
    rep.add("confounded_test", conf_res, time.perf_counter() - t0)

    t0 = section("5. CATE models on a held-out split")
    cate = {}
    for y in args.outcomes:
        cate[y] = {}
        for name, cls in ESTIMATORS.items():
            ts = time.perf_counter()
            model = cls(n_estimators=args.cf_trees, seed=args.seed) if name == "causalforest" else cls(seed=args.seed)
            res, test = model.fit_evaluate(sample, FEATURES, outcome_col=y)
            ev = res.evaluation
            cate[y][name] = {
                "method": res.method, "mean_cate": res.mean_cate, "n_train": res.n_train,
                "n_test": res.n_test, "qini_coefficient": ev["qini_coefficient"],
                "top_10pct": ev["top_10pct"], "top_25pct": ev["top_25pct"],
                "deciles": ev["deciles"], "feature_importance": res.feature_importance,
                "seconds": round(time.perf_counter() - ts, 1),
            }
            print(f"  {y:10s} {res.method:20s} Qini {ev['qini_coefficient']:+.3f}  "
                  f"top-25% observed lift {ev['top_25pct']['lift_vs_overall']:.2f}x "
                  f"[{ev['top_25pct']['ci_lower']:.5f}, {ev['top_25pct']['ci_upper']:.5f}]")
    rep.add("cate_models_holdout", cate, time.perf_counter() - t0)

    t0 = section("6. Refutation")
    dm = lambda d: difference_in_means(d, outcome_col=y0).ate  # noqa: E731
    def aipw_fn(d):
        cols = FEATURES + (["_random_cause"] if "_random_cause" in d else [])
        return aipw(d, cols, outcome_col=y0, propensity=p_treat).ate
    refute = {
        "outcome": y0,
        "placebo_treatment": placebo_treatment(dm, sample, n_permutations=50, seed=args.seed),
        "random_common_cause": random_common_cause(aipw_fn, sample, seed=args.seed),
        "subset_stability": subset_stability(dm, sample, n_subsets=10, seed=args.seed),
    }
    for k in ("placebo_treatment", "random_common_cause", "subset_stability"):
        print(f"  {k:22s} passed={refute[k]['passed']}")
    rep.add("refutation", refute, time.perf_counter() - t0)

    total = time.perf_counter() - start
    rep.data["total_seconds"] = round(total, 1)
    rep.out.write_text(json.dumps(rep.data, indent=2))
    print(f"\nDone in {total / 60:.1f} min → {rep.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

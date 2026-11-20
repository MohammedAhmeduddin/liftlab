"""
Benchmark every LiftLab estimator on the real Criteo uplift data.

Criteo is a randomized trial, so the difference in means on ALL rows is the
ground truth. Everything else is checked against it:

  1. RCT ground truth on the full data (conversion and visit)
  2. Observed uplift by f8 quartile on the full data (no model needed)
  3. AIPW and PSM on a model sample
  4. Confounded-sample test: bias a sample on purpose; can PSM / AIPW
     recover the RCT answer where the naive comparison fails?
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liftlab.causal.estimators import PropensityScoreMatching, aipw, difference_in_means  # noqa: E402
from liftlab.causal.evaluation import observed_uplift_by_group  # noqa: E402
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
    ap.add_argument("--psm-bootstrap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    rep = Report(Path(args.out), args)
    start = time.perf_counter()

    t0 = section("Loading data")
    full = load_criteo(path=args.data)
    rep.add("data", {"rows": len(full), "treated_share": float(full.treatment.mean()),
                     "conversion_rate": float(full.conversion.mean()),
                     "visit_rate": float(full.visit.mean())}, time.perf_counter() - t0)

    t0 = section("1. RCT ground truth on all rows")
    truth = {y: est_dict(difference_in_means(full, outcome_col=y)) for y in args.outcomes}
    for y, r in truth.items():
        print(f"  {y}: ATE {r['estimate']:.6f}  CI [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]  "
              f"relative lift {r['relative_lift']:+.1%}")
    rep.add("rct_ground_truth", truth, time.perf_counter() - t0)

    t0 = section("2. Observed uplift by f8 quartile (all rows)")
    f8_q = full["f8"].rank(method="first")
    f8_bins = (np.ceil(f8_q / len(full) * 4) - 1).astype(int).to_numpy()
    seg = {y: observed_uplift_by_group(full[y].to_numpy(float), full.treatment.to_numpy(int),
                                       f8_bins).to_dict(orient="records") for y in args.outcomes}
    for y, rows in seg.items():
        print(f"  {y}: " + "  ".join(f"Q{r['group'] + 1} {r['lift_vs_overall']:.2f}x" for r in rows))
    rep.add("f8_quartiles_observed", seg, time.perf_counter() - t0)

    sample = full.sample(frac=args.model_frac, random_state=args.seed).reset_index(drop=True)
    del full
    print(f"\nModel sample: {len(sample):,} rows ({args.model_frac:.0%})")

    t0 = section("3. AIPW and PSM on the model sample")
    p_treat = float(sample.treatment.mean())
    adj = {}
    for y in args.outcomes:
        adj[y] = {
            "difference_in_means": est_dict(difference_in_means(sample, outcome_col=y)),
            "aipw": est_dict(aipw(sample, FEATURES, outcome_col=y, propensity=p_treat)),
            "psm": est_dict(PropensityScoreMatching(n_bootstrap=args.psm_bootstrap,
                                                    seed=args.seed).estimate(sample, FEATURES, outcome_col=y)),
        }
        for k, r in adj[y].items():
            print(f"  {y:10s} {k:20s} {r['estimand']} {r['estimate']:.6f} "
                  f"CI [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]")
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
        "target": truth[y0]["estimate"],
        "naive_difference_in_means": est_dict(naive),
        "aipw_estimated_propensity": est_dict(aipw(conf, FEATURES, outcome_col=y0)),
        "psm": est_dict(PropensityScoreMatching(n_bootstrap=args.psm_bootstrap,
                                                seed=args.seed).estimate(conf, FEATURES, outcome_col=y0)),
    }
    for k in ("naive_difference_in_means", "aipw_estimated_propensity", "psm"):
        r = conf_res[k]
        print(f"  {k:28s} {r['estimate']:.6f}  CI [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]  "
              f"(target {conf_res['target']:.6f})")
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

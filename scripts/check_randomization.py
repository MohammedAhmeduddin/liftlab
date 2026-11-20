"""
Is treatment in the Criteo file really independent of the features?

The whole benchmark treats the difference in means on all rows as ground
truth. That is only valid if treatment does not depend on the features.
Criteo merged several advertiser tests and re-sampled them to one global
treatment ratio (v2), so it should hold, but this script checks it directly:

  1. Balance: standardized mean difference (treated vs control) per feature
  2. Treated share across value groups of each feature (should be flat)
  3. Classifier two-sample test: can a model predict treatment from the
     features on held-out rows? (AUC 0.5 = no)
  4. Difference in means vs AIPW on several independent samples: under a
     clean randomization they agree within noise

    python scripts/check_randomization.py
Writes results/randomization_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liftlab.causal.estimators import aipw, difference_in_means, standardized_mean_differences  # noqa: E402
from liftlab.causal.evaluation import value_bins  # noqa: E402
from liftlab.data.loaders import FEATURES, load_criteo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/criteo_uplift.csv")
    ap.add_argument("--out", default="results/randomization_check.json")
    ap.add_argument("--c2st-rows", type=int, default=2_000_000)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--sample-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {}

    df = load_criteo(sample_frac=1.0, path=args.data)
    p_treat = float(df["treatment"].mean())
    t = df["treatment"].to_numpy()

    print("\n── 1. Balance (|SMD|, < 0.1 is the usual bar) ──────────────")
    smd = standardized_mean_differences(df[t == 1], df[t == 0], FEATURES)
    for f, v in sorted(smd.items(), key=lambda kv: -kv[1]):
        print(f"  {f:4s} {v:.4f}")
    out["smd"] = smd

    print("\n── 2. Treated share by feature value group (overall "
          f"{p_treat:.3f}) ──")
    shares = {}
    for f in FEATURES:
        labels, edges = value_bins(df[f].to_numpy(), 10)
        g = df.groupby(labels)["treatment"].agg(["mean", "size"])
        shares[f] = {"min": float(g["mean"].min()), "max": float(g["mean"].max()),
                     "n_groups": int(len(g))}
        print(f"  {f:4s} {len(g):2d} groups  treated share {g['mean'].min():.3f} – {g['mean'].max():.3f}")
    out["treated_share_by_group"] = shares

    print("\n── 3. Classifier two-sample test ────────────────────────────")
    s = df.sample(n=min(args.c2st_rows, len(df)), random_state=args.seed)
    half = len(s) // 2
    X, y = s[FEATURES].to_numpy(dtype=float), s["treatment"].to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed).fit(X[:half], y[:half])
    auc = float(roc_auc_score(y[half:], clf.predict_proba(X[half:])[:, 1]))
    rng = np.random.default_rng(args.seed)
    null = [float(roc_auc_score(rng.permutation(y[half:]), clf.predict_proba(X[half:])[:, 1]))
            for _ in range(20)]
    print(f"  held-out AUC {auc:.4f}  (shuffled-label AUCs {min(null):.4f} – {max(null):.4f})")
    out["c2st"] = {"rows": len(s), "auc": auc, "null_auc_min": min(null), "null_auc_max": max(null)}

    print("\n── 4. Difference in means vs AIPW on independent samples ────")
    gaps = {}
    for outcome in ("visit", "conversion"):
        truth = difference_in_means(df, outcome_col=outcome).ate
        rows = []
        for i in range(args.n_samples):
            d = df.sample(frac=args.sample_frac, random_state=args.seed + 100 + i)
            dm = difference_in_means(d, outcome_col=outcome)
            ak = aipw(d, FEATURES, outcome_col=outcome, propensity=p_treat)
            ae = aipw(d, FEATURES, outcome_col=outcome)
            # Under randomization AIPW - DM has sd ~ sqrt(se_DM^2 - se_AIPW^2)
            var = dm.std_error**2 - ak.std_error**2
            z = float((ak.ate - dm.ate) / np.sqrt(var)) if var > 0 else None
            rows.append({"dm": dm.ate, "dm_se": dm.std_error, "aipw_known_p": ak.ate,
                         "aipw_se": ak.std_error, "aipw_estimated_p": ae.ate, "gap_z": z})
            zs = f"{z:+.2f}" if z is not None else "n/a"
            print(f"  {outcome:10s} #{i}  DM {dm.ate:.6f}  AIPW(known p) {ak.ate:.6f}  "
                  f"AIPW(est. p) {ae.ate:.6f}  gap z {zs}")
        gaps[outcome] = {"full_data_dm": truth, "samples": rows}
        print(f"  {outcome:10s} full-data DM {truth:.6f}")
    out["dm_vs_aipw"] = gaps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min → {args.out}")


if __name__ == "__main__":
    main()

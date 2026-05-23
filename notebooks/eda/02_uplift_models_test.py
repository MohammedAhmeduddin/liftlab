# notebooks/eda/02_uplift_models_test.py
import sys
import warnings
from pathlib import Path

# Suppress noisy sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from liftlab.data.loaders import load_criteo
from liftlab.causal.uplift import TLearnerUplift, XLearnerUplift, CausalForestUplift

print("=" * 60)
print("Milestone 3 — Uplift Models Test")
print("=" * 60)

# Small sample — CausalForest is slow on full data
df = load_criteo(sample_frac=0.01)
feature_cols = [f"f{i}" for i in range(12)]

print(f"\nWorking with {len(df):,} rows\n")

# ── 1. T-Learner ───────────────────────────────────────────
print("[1] T-Learner")
tl = TLearnerUplift()
tl_result = tl.fit_estimate(df, feature_cols)
print(f"✅ ATE={tl_result.ate:.6f} | CI=[{tl_result.ate_lower:.6f}, {tl_result.ate_upper:.6f}]")
print(f"   CATE range: [{tl_result.cate.min():.6f}, {tl_result.cate.max():.6f}]")

# ── 2. X-Learner ───────────────────────────────────────────
print("\n[2] X-Learner")
xl = XLearnerUplift()
xl_result = xl.fit_estimate(df, feature_cols)
print(f"✅ ATE={xl_result.ate:.6f} | CI=[{xl_result.ate_lower:.6f}, {xl_result.ate_upper:.6f}]")
print(f"   CATE range: [{xl_result.cate.min():.6f}, {xl_result.cate.max():.6f}]")

# ── 3. Causal Forest ───────────────────────────────────────
print("\n[3] CausalForest DML")
cf = CausalForestUplift(n_estimators=100)
cf_result = cf.fit_estimate(df, feature_cols)
print(f"✅ ATE={cf_result.ate:.6f} | CI=[{cf_result.ate_lower:.6f}, {cf_result.ate_upper:.6f}]")
print(f"   CATE range: [{cf_result.cate.min():.6f}, {cf_result.cate.max():.6f}]")

# ── 4. SHAP feature importance ─────────────────────────────
print("\n[4] SHAP Feature Importance (Treatment Effect Heterogeneity)")
if cf_result.shap_values is not None:
    importance = cf.feature_importance(feature_cols)
    print(importance.to_string(index=False))
    top_feature = importance.iloc[0]["feature"]
else:
    print("⚠️  SHAP values unavailable — using f0 as fallback segment feature")
    top_feature = "f0"

# ── 5. Segment analysis ────────────────────────────────────
print("\n[5] CATE by Segment (top feature)")
segments = cf_result.segment_effects(df, col=top_feature, n_bins=4)
print(f"Segmenting by '{top_feature}':")
print(segments.to_string(index=False))

# ── 6. Estimator comparison ────────────────────────────────
print("\n[6] Estimator Comparison")
print(f"{'Method':<16} {'ATE':>10} {'CI Lower':>10} {'CI Upper':>10}")
print("-" * 50)
for r in [tl_result, xl_result, cf_result]:
    print(f"{r.method:<16} {r.ate:>10.6f} {r.ate_lower:>10.6f} {r.ate_upper:>10.6f}")

print("\n" + "=" * 60)
print("Milestone 3 complete ✅")
print("=" * 60)

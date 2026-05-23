# notebooks/eda/01_causal_estimators_test.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from liftlab.data.loaders import load_criteo
from liftlab.causal.graph import build_causal_model, identify_estimand
from liftlab.causal.estimators import PropensityScoreMatching, DifferenceInDifferences

print("=" * 60)
print("Milestone 2 — Causal Estimators Test")
print("=" * 60)

# Use a small sample for speed
df = load_criteo(sample_frac=0.02)
feature_cols = [f"f{i}" for i in range(12)]

# ── 1. DoWhy causal graph ──────────────────────────────────
print("\n[1] DoWhy Causal Graph")
model = build_causal_model(df)
estimand = identify_estimand(model)
print("✅ Causal graph built and estimand identified")

# ── 2. PSM ────────────────────────────────────────────────
print("\n[2] Propensity Score Matching")
psm = PropensityScoreMatching(caliper=0.05, n_bootstrap=100)
psm_result = psm.estimate(df, feature_cols)
print(f"✅ PSM ATE     : {psm_result.ate:.6f}")
print(f"   95% CI     : [{psm_result.ci_lower:.6f}, {psm_result.ci_upper:.6f}]")
print(f"   p-value    : {psm_result.p_value:.4f}")
print(f"   Treated N  : {psm_result.n_treated:,}")
print(f"   Control N  : {psm_result.n_control:,}")

# ── 3. DiD (simulate pre/post periods) ────────────────────
print("\n[3] Difference-in-Differences")
# Simulate a pre/post split for demonstration
# In production this would come from real time-stamped data
df_did = df.copy()
np.random.seed(42)
df_did["time_period"] = np.random.binomial(1, 0.5, len(df_did))
# Add synthetic pre-period effect to make it realistic
df_did.loc[df_did["time_period"] == 0, "conversion"] = (
    df_did.loc[df_did["time_period"] == 0, "conversion"] * 0.8
).astype(int)

did = DifferenceInDifferences()
did_result = did.estimate(
    df_did,
    outcome_col="conversion",
    treatment_col="treatment",
    time_col="time_period",
    covariate_cols=feature_cols[:4],
)
print(f"✅ DiD ATE     : {did_result.ate:.6f}")
print(f"   95% CI     : [{did_result.ci_lower:.6f}, {did_result.ci_upper:.6f}]")
print(f"   p-value    : {did_result.p_value:.4f}")
print(f"   Parallel trends warning: {did_result.metadata['parallel_trends_warning']}")

print("\n" + "=" * 60)
print("Milestone 2 estimators working ✅")
print("=" * 60)

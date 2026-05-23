# notebooks/eda/04_drift_detection_test.py
"""
Milestone 5 — Drift Detection Test

Simulates three scenarios:
  1. No drift    — reference vs identical population
  2. Moderate    — reference vs slightly shifted population
  3. Severe      — reference vs heavily shifted population
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from liftlab.data.loaders import load_criteo
from liftlab.monitoring.drift import DriftDetector
from liftlab.monitoring.alerts import send_drift_alert

print("=" * 60)
print("Milestone 5 — Drift Detection Test")
print("=" * 60)

feature_cols = [f"f{i}" for i in range(12)]
df = load_criteo(sample_frac=0.05)

# Split into reference (first 60%) and current (last 40%)
split = int(len(df) * 0.6)
reference_df = df.iloc[:split].reset_index(drop=True)
current_df   = df.iloc[split:].reset_index(drop=True)

detector = DriftDetector(psi_threshold=0.20, mmd_threshold=0.05)

# ── Scenario 1: No drift ───────────────────────────────────
print("\n[1] Scenario: No Drift (same distribution)")
report_1 = detector.detect(reference_df, reference_df.copy(), feature_cols)
print(f"    Drift detected : {report_1.drift_detected}")
print(f"    Severity       : {report_1.severity}")
print(f"    MMD            : {report_1.mmd_score:.6f}")

# ── Scenario 2: Moderate drift ────────────────────────────
print("\n[2] Scenario: Moderate Drift (real train/test split)")
report_2 = detector.detect(reference_df, current_df, feature_cols)
print(f"    Drift detected : {report_2.drift_detected}")
print(f"    Severity       : {report_2.severity}")
print(f"    MMD            : {report_2.mmd_score:.6f}")
print(f"    PSI flagged    : {report_2.psi_flagged}")

# ── Scenario 3: Severe drift ──────────────────────────────
print("\n[3] Scenario: Severe Drift (heavily shifted population)")
shifted_df = current_df.copy()
rng = np.random.default_rng(42)
# Heavily shift half the features
for col in feature_cols[:6]:
    shifted_df[col] = shifted_df[col] + rng.normal(
        loc=shifted_df[col].mean() * 2,
        scale=shifted_df[col].std() * 3,
        size=len(shifted_df)
    )

report_3 = detector.detect(reference_df, shifted_df, feature_cols)
print(f"    Drift detected : {report_3.drift_detected}")
print(f"    Severity       : {report_3.severity}")
print(f"    MMD            : {report_3.mmd_score:.6f}")
print(f"    PSI flagged    : {report_3.psi_flagged}")
print(f"    Recommendation : {report_3.recommendation}")

# ── Alert test ────────────────────────────────────────────
print("\n[4] Alert Test (Slack not configured — should skip)")
sent = send_drift_alert(report_3, experiment_name="criteo_causalforest_v1")
print(f"    Alert sent: {sent}")

print("\n" + "=" * 60)
print("Milestone 5 complete ✅")
print("=" * 60)

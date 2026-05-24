# src/liftlab/monitoring/drift.py
"""
Drift detection for experiment populations.

Two methods:
  PSI  — Population Stability Index: detects distribution shift in a single feature
  MMD  — Maximum Mean Discrepancy: detects multivariate covariate shift

Thresholds (industry standard):
  PSI < 0.10  → no significant change
  PSI < 0.20  → moderate change, monitor
  PSI >= 0.20 → significant shift, trigger retraining

  MMD >= configured threshold → significant shift
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional


@dataclass
class DriftReport:
    """Results of a drift detection run."""
    reference_size: int
    current_size: int

    # PSI results per feature
    psi_scores: dict[str, float] = field(default_factory=dict)
    psi_flagged: list[str] = field(default_factory=list)   # features with PSI >= threshold

    # MMD result (multivariate)
    mmd_score: float = 0.0
    mmd_flagged: bool = False

    # Overall verdict
    drift_detected: bool = False
    severity: str = "none"    # none | moderate | severe
    recommendation: str = ""

    def summary(self) -> str:
        lines = [
            "Drift Report",
            f"  Reference N : {self.reference_size:,}",
            f"  Current N   : {self.current_size:,}",
            f"  MMD Score   : {self.mmd_score:.6f} {'⚠️' if self.mmd_flagged else '✅'}",
            f"  PSI Flagged : {self.psi_flagged if self.psi_flagged else 'none'}",
            f"  Drift       : {self.severity.upper()}",
            f"  Action      : {self.recommendation}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute PSI between reference and current distributions.

    PSI = Σ (current% - reference%) * ln(current% / reference%)

    Args:
        reference: 1D array from the reference/training population
        current:   1D array from the current/production population
        n_bins:    number of quantile bins
        epsilon:   small value to avoid log(0)

    Returns:
        PSI score (float)
    """
    # Build bins from reference distribution
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates from constant features

    # Compute bin frequencies
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Convert to proportions
    ref_pct = ref_counts / (len(reference) + epsilon)
    cur_pct = cur_counts / (len(current) + epsilon)

    # Add epsilon to avoid log(0)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_all_features(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.20,
) -> tuple[dict[str, float], list[str]]:
    """
    Compute PSI for every feature column.

    Returns:
        psi_scores: {feature: psi_value}
        flagged:    list of features with PSI >= threshold
    """
    psi_scores = {}
    flagged = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning(f"Column '{col}' missing — skipping PSI")
            continue

        psi = compute_psi(
            reference_df[col].dropna().values,
            current_df[col].dropna().values,
        )
        psi_scores[col] = psi

        status = "🔴 FLAGGED" if psi >= threshold else ("🟡 MONITOR" if psi >= 0.10 else "✅ OK")
        logger.info(f"PSI {col}: {psi:.4f} {status}")

        if psi >= threshold:
            flagged.append(col)

    return psi_scores, flagged


# ---------------------------------------------------------------------------
# MMD — Maximum Mean Discrepancy
# ---------------------------------------------------------------------------

def compute_mmd(
    reference: np.ndarray,
    current: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two multivariate samples.

    MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where k is an RBF kernel and x~P (reference), y~Q (current).

    A large MMD indicates the two samples come from different distributions.

    Args:
        reference: 2D array (n_ref, n_features)
        current:   2D array (n_cur, n_features)
        kernel:    only 'rbf' supported
        gamma:     RBF bandwidth (default: 1 / n_features)

    Returns:
        MMD score (float, >= 0)
    """
    # Cap sample sizes for computational efficiency
    max_samples = 2000
    rng = np.random.default_rng(42)

    if len(reference) > max_samples:
        idx = rng.choice(len(reference), max_samples, replace=False)
        reference = reference[idx]
    if len(current) > max_samples:
        idx = rng.choice(len(current), max_samples, replace=False)
        current = current[idx]

    if gamma is None:
        gamma = 1.0 / reference.shape[1]

    def rbf_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix K(X, Y)."""
        # ||x - y||² via expansion: ||x||² + ||y||² - 2<x,y>
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        XY = X @ Y.T
        sq_dists = XX + YY.T - 2 * XY
        return np.exp(-gamma * sq_dists)

    K_xx = rbf_kernel(reference, reference)
    K_yy = rbf_kernel(current, current)
    K_xy = rbf_kernel(reference, current)

    n = len(reference)
    m = len(current)

    # Unbiased MMD² estimator
    mmd2 = (
        (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
        + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
        - 2 * K_xy.mean()
    )

    return float(max(0.0, mmd2))  # clamp to 0 (can be slightly negative due to estimator)


# ---------------------------------------------------------------------------
# Full drift detection pipeline
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Orchestrates PSI + MMD drift detection between a reference
    population and a current population.
    """

    def __init__(self, psi_threshold: float = 0.20, mmd_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.mmd_threshold = mmd_threshold

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> DriftReport:
        """
        Run full drift detection pipeline.

        Args:
            reference_df: population used to train the original model
            current_df:   current production population
            feature_cols: features to monitor

        Returns:
            DriftReport with PSI scores, MMD score, and overall verdict
        """
        logger.info(
            f"Running drift detection | "
            f"reference={len(reference_df):,} | current={len(current_df):,}"
        )

        # PSI per feature
        psi_scores, psi_flagged = compute_psi_all_features(
            reference_df, current_df, feature_cols, self.psi_threshold
        )

        # MMD multivariate
        ref_X = reference_df[feature_cols].fillna(0).values
        cur_X = current_df[feature_cols].fillna(0).values
        mmd_score = compute_mmd(ref_X, cur_X)
        mmd_flagged = mmd_score >= self.mmd_threshold
        logger.info(
            f"MMD score: {mmd_score:.6f} "
            f"{'🔴 FLAGGED' if mmd_flagged else '✅ OK'} "
            f"(threshold={self.mmd_threshold})"
        )

        # Overall verdict
        drift_detected = bool(psi_flagged) or mmd_flagged
        n_flagged = len(psi_flagged)

        if not drift_detected:
            severity = "none"
            recommendation = "No action required — population is stable"
        elif n_flagged <= 2 and not mmd_flagged:
            severity = "moderate"
            recommendation = "Monitor closely — consider retraining within 1 week"
        else:
            severity = "severe"
            recommendation = "Trigger retraining immediately — past estimates may be invalid"

        report = DriftReport(
            reference_size=len(reference_df),
            current_size=len(current_df),
            psi_scores=psi_scores,
            psi_flagged=psi_flagged,
            mmd_score=mmd_score,
            mmd_flagged=mmd_flagged,
            drift_detected=drift_detected,
            severity=severity,
            recommendation=recommendation,
        )

        logger.info(f"\n{report.summary()}")
        return report

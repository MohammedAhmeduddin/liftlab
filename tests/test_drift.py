# tests/test_drift.py
"""Unit tests for drift detection."""
import numpy as np
import pandas as pd
import pytest
from liftlab.monitoring.drift import compute_psi, compute_mmd, DriftDetector


def make_df(n: int, shift: float = 0.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.standard_normal(n) + shift for i in range(4)}
    return pd.DataFrame(data)


def test_psi_zero_same_distribution():
    ref = np.random.default_rng(42).standard_normal(10000)
    psi = compute_psi(ref, ref.copy())
    assert psi < 0.01, f"PSI should be ~0 for same distribution, got {psi}"


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(10000)
    cur = rng.standard_normal(10000) + 5.0   # large shift
    psi = compute_psi(ref, cur)
    assert psi > 0.20, f"PSI should be large for shifted distribution, got {psi}"


def test_mmd_zero_same_distribution():
    ref = np.random.default_rng(42).standard_normal((500, 4))
    mmd = compute_mmd(ref, ref.copy())
    assert mmd < 0.01


def test_mmd_large_for_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((500, 4))
    cur = rng.standard_normal((500, 4)) + 3.0
    mmd = compute_mmd(ref, cur)
    assert mmd > 0.05


def test_drift_detector_no_drift():
    ref_df = make_df(1000)
    detector = DriftDetector(psi_threshold=0.20, mmd_threshold=0.05)
    report = detector.detect(ref_df, ref_df.copy(), [f"f{i}" for i in range(4)])
    assert not report.drift_detected
    assert report.severity == "none"


def test_drift_detector_severe_drift():
    rng = np.random.default_rng(42)
    ref_df = make_df(1000)
    cur_df = make_df(1000, shift=5.0)
    detector = DriftDetector(psi_threshold=0.20, mmd_threshold=0.05)
    report = detector.detect(ref_df, cur_df, [f"f{i}" for i in range(4)])
    assert report.drift_detected
    assert report.severity == "severe"

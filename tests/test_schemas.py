# tests/test_schemas.py
"""Pydantic schema validation tests."""
import pytest
from pydantic import ValidationError
from liftlab.api.schemas import ExperimentCreate, EstimateRequest


def test_experiment_create_valid():
    payload = {
        "name": "valid_exp",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0", "f1"],
        "estimator": "causalforest",
        "sample_frac": 0.05,
    }
    exp = ExperimentCreate(**payload)
    assert exp.name == "valid_exp"
    assert exp.estimator == "causalforest"


def test_experiment_create_rejects_empty_features():
    payload = {
        "name": "invalid",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": [],  # empty
        "estimator": "tlearner",
        "sample_frac": 0.01,
    }
    with pytest.raises(ValidationError):
        ExperimentCreate(**payload)


def test_experiment_create_rejects_invalid_estimator():
    payload = {
        "name": "invalid",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0"],
        "estimator": "not_a_real_estimator",
        "sample_frac": 0.01,
    }
    with pytest.raises(ValidationError):
        ExperimentCreate(**payload)


def test_experiment_create_rejects_invalid_sample_frac():
    payload = {
        "name": "invalid",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0"],
        "estimator": "psm",
        "sample_frac": 1.5,  # > 1.0
    }
    with pytest.raises(ValidationError):
        ExperimentCreate(**payload)


def test_estimate_request_valid():
    req = EstimateRequest(experiment_id="123e4567-e89b-12d3-a456-426614174000")
    assert req.run_async is False

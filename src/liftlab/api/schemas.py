# src/liftlab/api/schemas.py
"""
Pydantic request/response schemas for the LiftLab API.
"""
from __future__ import annotations
from uuid import UUID
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=255)
    description: Optional[str] = None
    treatment_col: str = Field(default="treatment")
    outcome_col: str = Field(default="conversion")
    feature_cols: list[str] = Field(..., min_length=1)
    # DiD is a library estimator for panel data; Criteo has no time dimension,
    # so it is not offered here.
    estimator: Literal[
        "difference_in_means", "aipw", "psm", "tlearner", "xlearner", "causalforest"
    ] = "difference_in_means"
    sample_frac: float = Field(default=0.1, ge=0.001, le=1.0)

    @field_validator("feature_cols")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("At least one feature column required")
        return v


class ExperimentResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    treatment_col: str
    outcome_col: str
    feature_cols: list[str]
    estimator: str
    sample_frac: float
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Estimation schemas
# ---------------------------------------------------------------------------

class EstimateRequest(BaseModel):
    experiment_id: UUID
    run_async: bool = False     # For future async support


class SegmentEffect(BaseModel):
    """Observed (not predicted) uplift in a held-out quartile of predicted effect."""
    group: int                      # 0 = highest predicted effect
    n: int
    uplift: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    lift_vs_overall: Optional[float]


class EstimateResponse(BaseModel):
    run_id: UUID
    experiment_id: UUID
    mlflow_run_id: Optional[str]
    estimator: str
    estimand: str                   # ATE or ATT

    # Average effect
    ate: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_width: Optional[float]
    p_value: Optional[float]

    # Sample info
    n_treated: int
    n_control: int
    n_total: int

    # Heterogeneous effects (CATE models only; evaluated on a held-out split)
    mean_cate: Optional[float] = None
    qini_coefficient: Optional[float] = None
    top_25pct_lift: Optional[float] = None
    feature_importance: Optional[dict[str, float]] = None
    segment_effects: Optional[list[SegmentEffect]] = None

    # Run info
    duration_seconds: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    mlflow: str

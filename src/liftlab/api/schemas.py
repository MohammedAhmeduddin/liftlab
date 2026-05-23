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
    estimator: Literal["psm", "did", "tlearner", "xlearner", "causalforest"] = "causalforest"
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
    bin: int
    mean_cate: float
    n: int
    feature_mean: float
    lift_vs_average: float


class EstimateResponse(BaseModel):
    run_id: UUID
    experiment_id: UUID
    mlflow_run_id: Optional[str]
    estimator: str

    # Core results
    ate: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    p_value: Optional[float]

    # Sample info
    n_treated: int
    n_control: int
    n_total: int

    # Heterogeneous effects
    shap_importance: Optional[dict[str, float]]
    segment_effects: Optional[list[SegmentEffect]]

    # Run info
    duration_seconds: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    mlflow: str

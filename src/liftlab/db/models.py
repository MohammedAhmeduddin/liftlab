# src/liftlab/db/models.py
"""
SQLAlchemy ORM models for experiment configs and causal results.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """
    Stores experiment configuration submitted by growth teams.
    One experiment → many EstimationRun results.
    """
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    treatment_col = Column(String(100), nullable=False, default="treatment")
    outcome_col = Column(String(100), nullable=False, default="conversion")
    feature_cols = Column(JSON, nullable=False)       # list[str]
    estimator = Column(String(50), nullable=False)    # psm|did|tlearner|xlearner|causalforest
    sample_frac = Column(Float, nullable=False, default=0.1)
    status = Column(String(50), nullable=False, default="pending")  # pending|running|done|failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    runs = relationship("EstimationRun", back_populates="experiment", cascade="all, delete")


class EstimationRun(Base):
    """
    Stores causal estimation results for a completed experiment run.
    """
    __tablename__ = "estimation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    mlflow_run_id = Column(String(255), nullable=True)

    # Core causal estimates
    ate = Column(Float, nullable=False)
    ci_lower = Column(Float, nullable=False)
    ci_upper = Column(Float, nullable=False)
    std_error = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)

    # Sample info
    n_treated = Column(Integer, nullable=True)
    n_control = Column(Integer, nullable=True)
    n_total = Column(Integer, nullable=True)

    # SHAP / segment results stored as JSON
    shap_importance = Column(JSON, nullable=True)     # {feature: importance}
    segment_effects = Column(JSON, nullable=True)     # list of segment dicts

    # Run metadata
    estimator = Column(String(50), nullable=False)
    duration_seconds = Column(Float, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="runs")

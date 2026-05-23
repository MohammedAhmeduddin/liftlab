# src/liftlab/api/routes/experiments.py
"""
CRUD endpoints for experiment configuration management.
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger

from liftlab.db.session import get_db_dependency
from liftlab.db.models import Experiment
from liftlab.api.schemas import ExperimentCreate, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db_dependency),
):
    """Register a new experiment configuration."""
    # Check for duplicate name
    existing = db.query(Experiment).filter(Experiment.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{payload.name}' already exists",
        )

    experiment = Experiment(**payload.model_dump())
    db.add(experiment)
    db.flush()
    db.refresh(experiment)
    logger.info(f"Created experiment '{experiment.name}' id={experiment.id}")
    return experiment


@router.get("/", response_model=list[ExperimentResponse])
def list_experiments(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db_dependency),
):
    """List all registered experiments."""
    return db.query(Experiment).offset(skip).limit(limit).all()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Get a single experiment by ID."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db_dependency),
):
    """Delete an experiment and all its runs."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(experiment)
    logger.info(f"Deleted experiment id={experiment_id}")

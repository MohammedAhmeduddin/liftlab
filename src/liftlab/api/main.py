# src/liftlab/api/main.py
"""
LiftLab FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from liftlab.config import get_settings
from liftlab.db.session import init_db
from liftlab.api.routes.experiments import router as experiments_router
from liftlab.api.routes.estimate import router as estimate_router
from liftlab.api.schemas import HealthResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables on startup."""
    logger.info("Starting LiftLab API...")
    init_db()
    yield
    logger.info("LiftLab API shutting down.")


app = FastAPI(
    title="LiftLab",
    description="Production causal inference engine for e-commerce promotion uplift",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiments_router)
app.include_router(estimate_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — instant, never hangs."""
    import socket
    from liftlab.db.session import engine
    from sqlalchemy import text

    # Check DB
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    # Check MLflow via raw socket — 2s timeout, never hangs
    try:
        s = socket.create_connection(("localhost", 5001), timeout=2)
        s.close()
        mlflow_status = "healthy"
    except OSError:
        mlflow_status = "unavailable (not required)"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        database=db_status,
        mlflow=mlflow_status,
    )

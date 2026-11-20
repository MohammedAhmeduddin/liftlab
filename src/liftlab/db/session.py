# src/liftlab/db/session.py
"""
SQLAlchemy session factory and database initialization.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from loguru import logger
from liftlab.config import get_settings
from liftlab.db.models import Base

settings = get_settings()

def normalize_url(url: str) -> str:
    """Use the psycopg (v3) driver explicitly. Plain ``postgresql://`` means
    psycopg2 on SQLAlchemy 2.0 but psycopg on 2.1; pinning it here makes
    the installed driver work on both."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


engine = create_engine(
    normalize_url(settings.database_url),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready ✅")


@contextmanager
def get_db() -> Session:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

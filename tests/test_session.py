# tests/test_session.py
"""Database session tests."""
from liftlab.db.session import init_db, get_db


def test_init_db_creates_tables():
    """init_db should create tables without error."""
    try:
        init_db()
    except Exception as e:
        pytest.fail(f"init_db raised {e}")


def test_get_db_context_manager():
    """get_db context manager should yield session."""
    from liftlab.db.session import get_db as get_db_context
    with get_db_context() as db:
        assert db is not None

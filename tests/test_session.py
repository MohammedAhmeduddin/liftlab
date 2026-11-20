# tests/test_session.py
"""Database session tests."""
import pytest

from liftlab.db.session import init_db


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


@pytest.mark.parametrize("url,expected", [
    ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("sqlite:///x.db", "sqlite:///x.db"),
])
def test_normalize_url(url, expected):
    from liftlab.db.session import normalize_url
    assert normalize_url(url) == expected

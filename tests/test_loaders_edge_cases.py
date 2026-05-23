# tests/test_loaders_edge_cases.py
"""Edge case tests for data loaders."""
import pytest
from liftlab.data.loaders import load_olist


def test_load_olist_returns_dict():
    """load_olist should return dict of DataFrames."""
    try:
        tables = load_olist()
        assert isinstance(tables, dict)
        assert len(tables) > 0
    except FileNotFoundError:
        pytest.skip("Olist data not available")


def test_load_olist_raises_if_missing():
    """Should raise FileNotFoundError if olist dir missing."""
    from liftlab.config import get_settings
    settings = get_settings()
    old_path = settings.olist_data_path
    settings.olist_data_path = "data/nonexistent/"
    
    with pytest.raises(FileNotFoundError):
        load_olist()
    
    settings.olist_data_path = old_path

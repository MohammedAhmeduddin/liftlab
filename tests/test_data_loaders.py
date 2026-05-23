# tests/test_data_loaders.py
"""Data loader tests."""
import pytest
import pandas as pd
from pathlib import Path
from liftlab.data.loaders import load_criteo


def test_load_criteo_returns_dataframe():
    """load_criteo should return a DataFrame."""
    df = load_criteo(sample_frac=0.001)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_criteo_has_required_columns():
    df = load_criteo(sample_frac=0.001)
    required = ["treatment", "conversion", "f0", "f1"]
    for col in required:
        assert col in df.columns


def test_load_criteo_sample_frac_works():
    df_small = load_criteo(sample_frac=0.001)
    df_large = load_criteo(sample_frac=0.01)
    assert len(df_large) > len(df_small)


def test_load_criteo_raises_if_file_missing():
    from liftlab.config import get_settings
    settings = get_settings()
    old_path = settings.criteo_data_path

    # Temporarily point to nonexistent file
    settings.criteo_data_path = "data/nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        load_criteo()

    # Restore
    settings.criteo_data_path = old_path

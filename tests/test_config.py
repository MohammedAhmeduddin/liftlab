# tests/test_config.py
"""Config and settings tests."""
import os
from liftlab.config import get_settings


def test_settings_loaded():
    settings = get_settings()
    assert settings.app_name == "LiftLab"
    assert settings.database_url is not None
    assert settings.mlflow_tracking_uri is not None


def test_settings_cached():
    """Settings should be cached via lru_cache."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2  # same object in memory


def test_psi_threshold_configurable():
    settings = get_settings()
    assert settings.psi_threshold == 0.2


def test_mmd_threshold_configurable():
    settings = get_settings()
    assert settings.mmd_threshold == 0.05

"""
Criteo Uplift v2.1 loader.

The CSV is ~3 GB / 13.98M rows. It is parsed once with compact dtypes and
cached next to it as Parquet (seconds to reload), instead of re-reading
the CSV on every request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from liftlab.config import get_settings

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
CRITEO_DTYPES = {**{f: "float32" for f in FEATURES}, **{c: "int8" for c in BINARY_COLUMNS}}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CRITEO_DTYPES if c not in df.columns]
    if missing:
        raise ValueError(f"Criteo data is missing columns: {missing}")
    for c in BINARY_COLUMNS:
        bad = set(pd.unique(df[c])) - {0, 1}
        if bad:
            raise ValueError(f"Column '{c}' must be 0/1, found {sorted(bad)[:5]}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Criteo features contain missing values")
    return df


def _cache_path(path: Path) -> Path:
    name = path.name.removesuffix(".gz").removesuffix(".csv")
    return path.with_name(f"{name}.parquet")


def load_criteo(
    sample_frac: float = 1.0,
    seed: int = 42,
    path: Optional[str | Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load (and optionally sample) the Criteo uplift data.

    Columns: f0-f11 (anonymized features), treatment (randomized, ~85% = 1),
    exposure, visit, conversion (primary outcome).
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    path = Path(path or get_settings().criteo_data_path)
    cache = _cache_path(path)

    if use_cache and cache.exists() and (not path.exists() or cache.stat().st_mtime >= path.stat().st_mtime):
        df = pd.read_parquet(cache)
        logger.info(f"Loaded {len(df):,} rows from cache {cache}")
    elif path.exists():
        logger.info(f"Parsing {path} (one-time; cached as Parquet afterwards)")
        df = pd.read_csv(path, usecols=list(CRITEO_DTYPES), dtype=CRITEO_DTYPES)
        if use_cache:
            try:
                df.to_parquet(cache, index=False)
                logger.info(f"Cached to {cache}")
            except Exception as e:  # cache is an optimisation only
                logger.warning(f"Could not write Parquet cache: {e}")
    else:
        raise FileNotFoundError(
            f"Criteo dataset not found at {path}. Download criteo-uplift-v2.1.csv.gz from "
            "https://ailab.criteo.com/criteo-uplift-prediction-dataset/ and gunzip it there."
        )

    df = _validate(df)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    logger.info(f"Using {len(df):,} rows | treated {df['treatment'].mean():.3f} | "
                f"conversion {df['conversion'].mean():.4f}")
    return df

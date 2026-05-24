# src/liftlab/data/loaders.py
import pandas as pd
from pathlib import Path
from loguru import logger
from liftlab.config import get_settings

settings = get_settings()


def load_criteo(sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Load the Criteo Uplift v2.1 dataset.

    Columns:
        f0-f11  : 12 anonymized features
        treatment: 1 = treated, 0 = control
        exposure : whether the user was exposed
        visit    : visited the site (intermediate outcome)
        conversion: purchased (primary outcome)
    """
    path = Path(settings.criteo_data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Criteo dataset not found at {path}. "
            "Download from https://ailab.criteo.com/ressources/criteo-uplift-prediction-dataset/"
        )

    logger.info(f"Loading Criteo dataset from {path}")
    df = pd.read_csv(path)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} rows | treatment rate: {df['treatment'].mean():.3f} | "
                f"conversion rate: {df['conversion'].mean():.4f}")
    return df


def load_olist() -> dict[str, pd.DataFrame]:
    """
    Load all Olist CSVs into a dict of DataFrames keyed by table name.
    """
    olist_path = Path(settings.olist_data_path)
    if not olist_path.exists():
        raise FileNotFoundError(f"Olist data directory not found at {olist_path}")

    tables = {}
    for csv_file in olist_path.glob("*.csv"):
        table_name = csv_file.stem.replace("olist_", "").replace("_dataset", "")
        tables[table_name] = pd.read_csv(csv_file)
        logger.info(f"Loaded olist.{table_name}: {len(tables[table_name]):,} rows")

    return tables

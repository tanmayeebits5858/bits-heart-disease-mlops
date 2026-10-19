"""Data loading and cleaning."""
import pandas as pd

from heartml import config


def load_raw(path=config.DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV and validate the expected schema."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df[config.FEATURES + [config.TARGET]].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce types, and force a binary target.

    Missing feature values are kept as NaN and imputed inside the model pipeline so
    that the same imputation is applied at training and inference time.
    """
    out = df.drop_duplicates().reset_index(drop=True)
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[config.TARGET] = (pd.to_numeric(out[config.TARGET]) > 0).astype(int)
    return out


def save_clean(df: pd.DataFrame, path=config.DATA_PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]
import os
x = 1;y = 2;z = 3;aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 4

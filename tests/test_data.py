import numpy as np
import pandas as pd
import pytest

from heartml import config, data


def make_df(n=20):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(0, 3, n) for c in config.CATEGORICAL_FEATURES})
    for c in config.NUMERIC_FEATURES:
        df[c] = rng.normal(100, 10, n)
    df[config.TARGET] = rng.integers(0, 5, n)
    return df


def test_clean_binarizes_target():
    out = data.clean(make_df())
    assert set(out[config.TARGET].unique()) <= {0, 1}


def test_clean_removes_duplicates():
    df = make_df()
    out = data.clean(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    assert len(out) == len(df)


def test_clean_coerces_bad_values_to_nan():
    df = make_df()
    df["chol"] = df["chol"].astype(object)
    df.loc[0, "chol"] = "?"
    out = data.clean(df)
    assert out["chol"].isna().sum() == 1


def test_split_xy_columns():
    X, y = data.split_xy(data.clean(make_df()))
    assert list(X.columns) == config.FEATURES
    assert y.name == config.TARGET


def test_load_raw_validates_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        data.load_raw(bad)


def test_load_raw_real_file():
    if not config.DATA_RAW.exists():
        pytest.skip("run scripts/download_data.py first")
    df = data.load_raw()
    assert len(df) > 250
    assert df[config.TARGET].isin([0, 1]).all()

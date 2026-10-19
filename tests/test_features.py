import numpy as np
import pandas as pd

from heartml import config
from heartml.features import build_preprocessor
from tests.test_data import make_df


def test_preprocessor_output_is_finite_and_scaled():
    X = make_df(60)[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert np.isfinite(out).all()
    assert abs(out[:, : len(config.NUMERIC_FEATURES)].mean()) < 1e-6


def test_preprocessor_imputes_missing():
    X = make_df(60)[config.FEATURES]
    X.loc[0, "age"] = np.nan
    X.loc[1, "cp"] = np.nan
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert np.isfinite(out).all()


def test_preprocessor_ignores_unseen_category():
    X = make_df(60)[config.FEATURES]
    prep = build_preprocessor().fit(X)
    new = X.iloc[[0]].copy()
    new["cp"] = 99
    out = prep.transform(new)
    assert out.shape[0] == 1


def test_preprocessor_is_deterministic():
    X = make_df(60)[config.FEATURES]
    a = build_preprocessor().fit_transform(X)
    b = build_preprocessor().fit_transform(X)
    a = a.toarray() if hasattr(a, "toarray") else a
    b = b.toarray() if hasattr(b, "toarray") else b
    assert pd.DataFrame(a).equals(pd.DataFrame(b))

import numpy as np
import pytest

from heartml import config
from heartml.train import candidate_models, evaluate, make_pipeline
from tests.test_data import make_df


@pytest.fixture(scope="module")
def synthetic():
    df = make_df(120)
    df[config.TARGET] = (df["thalach"] > df["thalach"].median()).astype(int)
    return df[config.FEATURES], df[config.TARGET]


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_models_fit_and_predict_probabilities(name, synthetic):
    X, y = synthetic
    estimator, _ = candidate_models()[name]
    model = make_pipeline(estimator).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_evaluate_returns_all_metrics(synthetic):
    X, y = synthetic
    model = make_pipeline(candidate_models()["logistic_regression"][0]).fit(X, y)
    metrics = evaluate(model, X, y)
    assert set(metrics) == {"test_accuracy", "test_precision", "test_recall",
                            "test_f1", "test_roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_saved_model_meets_quality_floor():
    if not config.MODEL_PATH.exists() or not config.DATA_RAW.exists():
        pytest.skip("train the model first")
    import joblib

    from heartml import data

    X, y = data.split_xy(data.clean(data.load_raw()))
    model = joblib.load(config.MODEL_PATH)
    assert evaluate(model, X, y)["test_roc_auc"] > 0.8

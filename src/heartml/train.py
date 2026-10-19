"""Train, tune, evaluate and package classifiers with MLflow tracking."""
import json
import platform
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from heartml import config, data  # noqa: E402
from heartml.features import build_preprocessor  # noqa: E402

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidate_models() -> dict:
    """Model name -> (estimator, hyper-parameter grid)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_plots(model, X_test, y_test, name: str, tmp: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC curve - {name}")
    fig.tight_layout()
    fig.savefig(tmp / f"roc_{name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(tmp / f"cm_{name}.png", dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(tmp / f"roc_{name}.png"), "plots")
    mlflow.log_artifact(str(tmp / f"cm_{name}.png"), "plots")


def run_training() -> dict:
    df = data.clean(data.load_raw())
    X, y = data.split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=config.MLFLOW_ARTIFACTS)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    results, fitted = {}, {}
    for name, (estimator, grid) in candidate_models().items():
        with mlflow.start_run(run_name=name), tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            search = GridSearchCV(make_pipeline(estimator), grid, scoring="roc_auc",
                                  cv=cv, n_jobs=1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Cross-validated metrics of the tuned model on the training split
            cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            cv_metrics = {}
            for key in SCORING:
                vals = cvres[f"test_{key}"]
                cv_metrics[f"cv_{key}_mean"] = float(vals.mean())
                cv_metrics[f"cv_{key}_std"] = float(vals.std())

            hold = {k: float(v) for k, v in evaluate(best, X_test, y_test).items()}

            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({"model": name, "cv_folds": config.CV_FOLDS,
                               "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
                               "n_train": len(X_train), "n_test": len(X_test)})
            mlflow.log_metrics({**cv_metrics, **hold})

            grid_df = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ]
            grid_df.to_csv(tmp / "grid_search_results.csv", index=False)
            mlflow.log_artifact(str(tmp / "grid_search_results.csv"))
            log_plots(best, X_test, y_test, name, tmp)
            sig = infer_signature(X_train, best.predict_proba(X_train)[:, 1])
            mlflow.sklearn.log_model(best, name="model", signature=sig,
                                     input_example=X_train.head(3),
                                     serialization_format="cloudpickle")

            results[name] = {
                "best_params": dict(search.best_params_),
                **cv_metrics, **hold,
            }
            fitted[name] = best

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    final = fitted[best_name]

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_PATH)
    metadata = {
        "selected_model": best_name,
        "selection_criterion": "highest mean cross-validated ROC-AUC",
        "features": config.FEATURES,
        "metrics": results[best_name],
        "all_results": results,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "mlflow": mlflow.__version__, "pandas": pd.__version__},
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))

    with mlflow.start_run(run_name="final_selected_model"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({k: v for k, v in results[best_name].items()
                            if isinstance(v, float)})
        mlflow.log_artifact(str(config.METADATA_PATH))
        mlflow.log_artifact(str(config.MODEL_PATH))
    return metadata


if __name__ == "__main__":
    meta = run_training()
    summary = {k: {m: round(v, 4) for m, v in r.items() if isinstance(v, float)}
               for k, r in meta["all_results"].items()}
    print(json.dumps({"selected_model": meta["selected_model"], "all_results": summary},
                     indent=2))

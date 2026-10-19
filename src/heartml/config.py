"""Central configuration: paths, schema and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FIGURES_DIR = ROOT / "reports" / "figures"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS = str(ROOT / "mlartifacts")
EXPERIMENT_NAME = "heart-disease-classification"

TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

"""Download the Heart Disease (Cleveland) dataset to data/raw/heart.csv.

Order of sources:
1. UCI Machine Learning Repository (raw processed.cleveland.data), converted to CSV.
2. GitHub mirror of the cleaned 303-row Cleveland CSV (used when UCI is unreachable).

Usage: python scripts/download_data.py [--source auto|uci|mirror]
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "heart.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
MIRROR_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def fetch(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig")


def from_uci() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(fetch(UCI_URL)), header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)  # 0 = absence, 1-4 = presence
    return df


def from_mirror() -> pd.DataFrame:
    """Load the mirror and align its label with UCI semantics (1 = disease present).

    The mirror's target is inverted relative to UCI: target=1 correlates negatively
    with exercise-induced angina and ST depression and positively with maximum heart
    rate, which is the pattern of healthy patients. The label is flipped here.
    """
    df = pd.read_csv(io.StringIO(fetch(MIRROR_URL)))
    df["target"] = 1 - df["target"]
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "uci", "mirror"], default="auto")
    args = parser.parse_args()

    df = None
    if args.source in ("auto", "uci"):
        try:
            df = from_uci()
            print("Downloaded from UCI.")
        except Exception as exc:  # noqa: BLE001
            print(f"UCI download failed: {exc}")
            if args.source == "uci":
                return 1
    if df is None:
        df = from_mirror()
        print("Downloaded from GitHub mirror.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

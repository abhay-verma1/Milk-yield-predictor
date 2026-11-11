"""Data loading, validation and the sklearn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.drop_duplicates().dropna(subset=[config.TARGET])


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC),
        ("cat", categorical, config.CATEGORICAL),
    ])


def validate_input(record: dict) -> dict:
    """Validate a single prediction request; raise ValueError on bad input."""
    if record.get("breed") not in config.BREEDS:
        raise ValueError(f"breed must be one of {config.BREEDS}")
    clean = {"breed": record["breed"]}
    for col, (lo, hi) in config.RANGES.items():
        if col not in record:
            raise ValueError(f"Missing field: {col}")
        try:
            val = float(record[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number") from None
        if not lo <= val <= hi:
            raise ValueError(f"{col} must be between {lo} and {hi} (got {val})")
        clean[col] = val
    return clean

"""Predict daily milk yield for one cow.

Example:
  python -m src.predict --breed "Holstein Friesian" --lactation_no 3 \
      --days_in_milk 60 --body_weight_kg 600 --concentrate_kg 6 \
      --green_fodder_kg 28 --avg_temp_c 25 --humidity_pct 55
"""
import argparse

import joblib
import pandas as pd

from . import config
from .preprocess import validate_input


def predict_yield(record: dict) -> float:
    """Return predicted milk yield in litres/day for a single cow record."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")
    clean = validate_input(record)
    model = joblib.load(config.MODEL_PATH)
    return round(float(model.predict(pd.DataFrame([clean])[config.FEATURES])[0]), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict milk yield (litres/day)")
    ap.add_argument("--breed", required=True, choices=config.BREEDS)
    for col in config.NUMERIC:
        ap.add_argument(f"--{col}", required=True, type=float)
    args = vars(ap.parse_args())
    try:
        print(f"Predicted milk yield: {predict_yield(args)} litres/day")
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()

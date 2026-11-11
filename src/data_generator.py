"""Generate a realistic synthetic dairy dataset.

Real farm data is rarely public, so we simulate it using known dairy science:
  * Wood's lactation curve:  y = a * t^b * exp(-c * t)
  * Breed-specific potential yield
  * Heat stress via Temperature-Humidity Index (THI)
  * Feed response with diminishing returns
Run:  python -m src.data_generator --rows 5000
"""
import argparse

import numpy as np
import pandas as pd

from . import config

# breed: (peak-yield multiplier, typical body weight kg)
BREED_PROFILE = {
    "Holstein Friesian": (1.35, 600),
    "Jersey": (1.00, 400),
    "Gir": (0.60, 420),
    "Sahiwal": (0.70, 380),
    "Crossbred": (0.95, 450),
}


def generate(rows: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    breed = rng.choice(config.BREEDS, size=rows, p=[0.25, 0.2, 0.15, 0.15, 0.25])
    mult = np.array([BREED_PROFILE[b][0] for b in breed])
    bw_mean = np.array([BREED_PROFILE[b][1] for b in breed])

    lactation_no = rng.integers(1, 9, rows)
    dim = rng.integers(5, 330, rows)
    body_weight = np.clip(rng.normal(bw_mean, 40), 250, 800)
    concentrate = np.clip(rng.normal(5, 1.8, rows), 0, 15)
    fodder = np.clip(rng.normal(25, 8, rows), 0, 60)
    temp = np.clip(rng.normal(27, 7, rows), -5, 48)
    humidity = np.clip(rng.normal(60, 18, rows), 10, 100)

    # Wood's curve (peak around week 6-8)
    b, c = 0.2, 0.0045
    wood = (dim ** b) * np.exp(-c * dim)
    base = 8.5 * mult * wood

    # Parity effect: first-calvers give less, peak at 3rd-5th lactation
    parity = (1 - 0.15 * (lactation_no == 1)
              + 0.05 * np.isin(lactation_no, [3, 4, 5])
              - 0.04 * (lactation_no >= 7))

    # Feed effect with diminishing returns
    feed = 1 + 0.35 * (1 - np.exp(-concentrate / 4)) + 0.004 * (fodder - 25)

    # Heat stress: THI above 72 reduces yield
    thi = (1.8 * temp + 32) - (0.55 - 0.0055 * humidity) * (1.8 * temp - 26)
    heat = np.clip(1 - 0.025 * np.clip(thi - 72, 0, None), 0.5, 1.0)

    milk = base * parity * feed * heat + rng.normal(0, 0.9, rows)
    milk = np.clip(milk, 0.5, None).round(2)

    return pd.DataFrame({
        "breed": breed,
        "lactation_no": lactation_no,
        "days_in_milk": dim,
        "body_weight_kg": body_weight.round(1),
        "concentrate_kg": concentrate.round(2),
        "green_fodder_kg": fodder.round(2),
        "avg_temp_c": temp.round(1),
        "humidity_pct": humidity.round(1),
        config.TARGET: milk,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", default=str(config.DATA_PATH))
    args = ap.parse_args()
    df = generate(args.rows)
    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

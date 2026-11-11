"""Central configuration for the Milk Yield Predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dairy_data.csv"
MODEL_PATH = ROOT / "models" / "milk_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
PLOT_PATH = ROOT / "reports" / "actual_vs_predicted.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "milk_yield_l_per_day"
CATEGORICAL = ["breed"]
NUMERIC = [
    "lactation_no",
    "days_in_milk",
    "body_weight_kg",
    "concentrate_kg",
    "green_fodder_kg",
    "avg_temp_c",
    "humidity_pct",
]
FEATURES = CATEGORICAL + NUMERIC

BREEDS = ["Holstein Friesian", "Jersey", "Gir", "Sahiwal", "Crossbred"]

# Valid ranges used for input validation (min, max)
RANGES = {
    "lactation_no": (1, 10),
    "days_in_milk": (1, 400),
    "body_weight_kg": (250, 800),
    "concentrate_kg": (0, 15),
    "green_fodder_kg": (0, 60),
    "avg_temp_c": (-5, 50),
    "humidity_pct": (5, 100),
}

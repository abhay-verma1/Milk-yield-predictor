"""Train several regressors, pick the best by cross-validated RMSE, and save it.

Run:  python -m src.train
"""
import json

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data_generator import generate
from .preprocess import build_preprocessor, load_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def candidates():
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=config.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=config.RANDOM_STATE),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
        "R2": round(float(r2_score(y, pred)), 3),
    }


def main() -> None:
    if not config.DATA_PATH.exists():
        print("No dataset found - generating synthetic data...")
        config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.DATA_PATH, index=False)

    df = load_data()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, reg in candidates().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", reg)])
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=5,
                                   scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        results[name] = {"cv_RMSE": round(float(cv_rmse), 3), **evaluate(pipe, X_te, y_te)}
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best = min(results, key=lambda n: results[n]["cv_RMSE"])
    print(f"\nBest model: {best}")

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(
        {"best_model": best, "train_rows": len(X_tr), "test_rows": len(X_te),
         "results": results}, indent=2))

    pred = fitted[best].predict(X_te)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_te, pred, s=8, alpha=0.5)
    lim = [0, max(y_te.max(), pred.max()) + 1]
    plt.plot(lim, lim, "r--", label="perfect prediction")
    plt.xlabel("Actual milk yield (L/day)")
    plt.ylabel("Predicted milk yield (L/day)")
    plt.title(f"{best}: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=120)
    print(f"Saved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

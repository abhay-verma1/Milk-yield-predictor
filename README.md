# Milk Yield Predictor (Dairy ML Project)

Predicts a cow's daily milk production (litres/day) from breed, lactation stage,
feed, body weight and weather using machine learning.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python -m src.data_generator --rows 5000   # 1. create the dataset (data/dairy_data.csv)
python -m src.train                        # 2. train 3 models, save the best
python -m src.predict --breed "Holstein Friesian" --lactation_no 3 \
    --days_in_milk 60 --body_weight_kg 600 --concentrate_kg 6 \
    --green_fodder_kg 28 --avg_temp_c 25 --humidity_pct 55   # 3. predict
python -m pytest -q                        # run tests
streamlit run app.py                       # web interface (opens in browser)
```

Using your own farm data: save a CSV with the same columns as `data/dairy_data.csv`
(see `src/config.py`) and run `python -m src.train`.

## Structure
```
src/config.py          settings, feature lists, valid ranges
src/data_generator.py  synthetic data (Wood's lactation curve + heat stress)
src/preprocess.py      loading, validation, sklearn preprocessing
src/train.py           trains Ridge / Random Forest / Gradient Boosting, saves best
src/predict.py         command-line prediction
app.py                 Streamlit web frontend
tests/                 pytest unit + end-to-end tests
reports/               metrics.json and actual-vs-predicted plot
```

## Results (test set, 1000 rows)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge Regression | 1.574 | 2.057 | 0.830 |
| Random Forest | 0.994 | 1.277 | 0.934 |
| **Gradient Boosting** | **0.867** | **1.100** | **0.951** |

import pytest

from src import config
from src.data_generator import generate
from src.preprocess import build_preprocessor, validate_input

GOOD = dict(breed="Jersey", lactation_no=2, days_in_milk=60, body_weight_kg=400,
            concentrate_kg=5, green_fodder_kg=25, avg_temp_c=25, humidity_pct=60)


def test_generator_shape_and_columns():
    df = generate(200)
    assert len(df) == 200
    assert set(config.FEATURES + [config.TARGET]) <= set(df.columns)
    assert (df[config.TARGET] > 0).all()


def test_preprocessor_output_rows():
    df = generate(100)
    out = build_preprocessor().fit_transform(df[config.FEATURES])
    assert out.shape[0] == 100


def test_validate_accepts_good_input():
    assert validate_input(GOOD)["breed"] == "Jersey"


@pytest.mark.parametrize("field,value", [
    ("breed", "Unicorn"), ("days_in_milk", 999), ("humidity_pct", -3),
    ("body_weight_kg", "abc"),
])
def test_validate_rejects_bad_input(field, value):
    with pytest.raises(ValueError):
        validate_input({**GOOD, field: value})


def test_validate_missing_field():
    bad = {k: v for k, v in GOOD.items() if k != "avg_temp_c"}
    with pytest.raises(ValueError):
        validate_input(bad)


def test_end_to_end_prediction():
    from src.predict import predict_yield
    from src.train import main as train_main
    train_main()
    holstein = predict_yield({**GOOD, "breed": "Holstein Friesian", "body_weight_kg": 600})
    gir = predict_yield({**GOOD, "breed": "Gir", "body_weight_kg": 420})
    assert holstein > gir > 0

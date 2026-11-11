"""Streamlit frontend for the Milk Yield Predictor.

Run:  streamlit run app.py
"""
import streamlit as st

from src import config
from src.predict import predict_yield

st.set_page_config(page_title="Milk Yield Predictor", page_icon="🐄", layout="centered")
st.title("🐄 Milk Yield Predictor")
st.caption("Estimate a cow's daily milk production from breed, feed and weather.")

if not config.MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

R = config.RANGES
with st.form("cow_form"):
    breed = st.selectbox("Breed", config.BREEDS)
    c1, c2 = st.columns(2)
    with c1:
        lactation_no = st.number_input("Lactation number", *R["lactation_no"], value=3, step=1)
        days_in_milk = st.number_input("Days in milk", *R["days_in_milk"], value=60, step=1)
        body_weight_kg = st.number_input("Body weight (kg)", *R["body_weight_kg"], value=500, step=10)
        concentrate_kg = st.number_input("Concentrate feed (kg/day)", 0.0, 15.0, 5.0, 0.5)
    with c2:
        green_fodder_kg = st.number_input("Green fodder (kg/day)", 0.0, 60.0, 25.0, 1.0)
        avg_temp_c = st.number_input("Average temperature (°C)", -5.0, 50.0, 27.0, 0.5)
        humidity_pct = st.number_input("Humidity (%)", 5.0, 100.0, 60.0, 1.0)
    submitted = st.form_submit_button("Predict milk yield", type="primary")

if submitted:
    record = dict(breed=breed, lactation_no=lactation_no, days_in_milk=days_in_milk,
                  body_weight_kg=body_weight_kg, concentrate_kg=concentrate_kg,
                  green_fodder_kg=green_fodder_kg, avg_temp_c=avg_temp_c,
                  humidity_pct=humidity_pct)
    try:
        result = predict_yield(record)
        st.metric("Predicted milk yield", f"{result} litres/day")
        st.caption(f"About {round(result * 30)} litres per 30 days at this rate.")
    except ValueError as e:
        st.error(str(e))

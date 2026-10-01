"""AutoWorth AI — Streamlit Smart Deal Advisor."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="AutoWorth AI", page_icon="🚗", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ARTIFACTS / "autoworth_bundle.joblib")
    return bundle


def deal_rating(predicted: float, seller: float) -> tuple[str, str, str]:
    if predicted <= 0:
        return "Unknown", "gray", "Predicted price is not valid."
    pct = (seller - predicted) / predicted * 100
    if pct < -10:
        return "GREAT DEAL", "green", "More than 10% cheaper than the predicted market price."
    if pct < -5:
        return "GOOD DEAL", "green", "5–10% cheaper than the predicted market price."
    if pct <= 5:
        return "FAIR PRICE", "gold", "Within about ±5% of the predicted market price."
    if pct <= 10:
        return "SLIGHTLY OVERPRICED", "orange", "5–10% more expensive than the predicted market price."
    return "OVERPRICED", "red", "More than 10% above the predicted market price."


def main():
    if not (ARTIFACTS / "autoworth_bundle.joblib").exists():
        st.error(
            "Model artifacts were not found. Run `python train_and_export.py` first, "
            "or execute the AutoWorth_AI notebook through the export cell."
        )
        return

    bundle = load_artifacts()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_order = bundle["feature_order"]
    choices = bundle["choices"]
    reference_year = bundle["reference_year"]

    st.title("AutoWorth AI")
    st.caption("Used car fair-market price estimator and smart deal advisor")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Car details")
        make = st.selectbox("Make", choices["makes"])
        models = choices["models_by_make"].get(make, choices["models"])
        model_name = st.selectbox("Model", models)
        year = st.slider("Year", int(choices["year_min"]), int(choices["year_max"]), 2017)
        mileage = st.number_input("Mileage", min_value=0, max_value=250000, value=25000, step=500)
        transmission = st.selectbox("Transmission", choices["transmissions"])
        fuel_type = st.selectbox("Fuel Type", choices["fuel_types"])
        engine_size = st.number_input(
            "Engine Size (litres)", min_value=0.5, max_value=6.6, value=2.0, step=0.1
        )
        tax = st.number_input("Tax (£)", min_value=0, max_value=600, value=145, step=5)
        mpg = st.number_input("MPG", min_value=1.0, max_value=300.0, value=50.0, step=0.5)
        seller_price = st.number_input("Seller Price (£)", min_value=500, max_value=150000, value=17000, step=100)
        submit = st.button("Estimate market price", type="primary")

    if not submit:
        st.info("Enter the listing details and click **Estimate market price**.")
        return

    car_age = max(reference_year - int(year), 0)
    mileage_per_year = mileage / max(car_age, 1)
    premium_makes = {"Audi", "BMW", "Mercedes"}
    is_premium = int(make in premium_makes)
    engine_efficiency = mpg / max(engine_size, 0.1)

    row = pd.DataFrame(
        [
            {
                "make": make,
                "model": model_name,
                "year": year,
                "mileage": mileage,
                "transmission": transmission,
                "fuelType": fuel_type,
                "tax": tax,
                "mpg": mpg,
                "engineSize": engine_size,
                "car_age": car_age,
                "mileage_per_year": mileage_per_year,
                "is_premium": is_premium,
                "engine_efficiency": engine_efficiency,
            }
        ]
    )[feature_order]

    X = preprocessor.transform(row)
    predicted = float(model.predict(X)[0])
    difference = seller_price - predicted
    rating, color, meaning = deal_rating(predicted, seller_price)

    with right:
        st.subheader("Deal assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated market price", f"£{predicted:,.0f}")
        m2.metric("Seller price", f"£{seller_price:,.0f}")
        cheaper = difference < 0
        m3.metric(
            "Difference",
            f"£{abs(difference):,.0f} {'cheaper' if cheaper else 'more expensive'}",
        )
        st.markdown(
            f"<div style='padding:1rem;border-radius:12px;background:{color};color:black;font-weight:700;text-align:center'>"
            f"{rating}</div>",
            unsafe_allow_html=True,
        )
        st.write(meaning)
        st.caption(
            f"Reference year for car age is {reference_year}. "
            "The rating compares the seller asking price with the model's predicted market value."
        )


if __name__ == "__main__":
    main()

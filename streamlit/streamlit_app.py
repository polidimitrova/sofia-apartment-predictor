import sys
sys.path.append("..")

import streamlit as st
import pandas as pd

from src.visualization import plot_dashboard
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.data_loader import load_data
from src.preprocessing import prepare_features
from src.model import train_model


st.set_page_config(
    page_title="Sofia Apartment Predictor",
    layout="wide"
)

st.markdown("""
<style>

/* GLOBAL */
html, body, [data-testid="stAppViewContainer"], .main, header, #root {
    background-color: #061224 !important;
    color: #f1f5f9 !important;
}


/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #081a33 !important;
    border-right: 1px solid #14233e;
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}


/* NUMBER INPUT */
div[data-baseweb="input"] {
    background-color: #0e2344 !important;
    border: 1px solid #22345c !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    background-color: #0e2344 !important;
    color: white !important;
}


/* REAL +/- FIX */
[data-testid="stNumberInput"] button {

    background-color: #0e2344 !important;

    color: white !important;

    border: none !important;

    box-shadow: none !important;
}

[data-testid="stNumberInput"] button svg {

    fill: white !important;

    stroke: white !important;

    color: white !important;
}

[data-testid="stNumberInput"] button:hover {

    background-color: #1d4ed8 !important;
}


/* SELECT */
div[data-baseweb="select"] {
    background-color: #0e2344 !important;
    border: 1px solid #22345c !important;
    border-radius: 8px !important;
}

/* DROPDOWNS */
[data-baseweb="select"] {
    background: #0e2344 !important;
    border: 1px solid #22345c !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] * {
    color: white !important;
    background: #0e2344 !important;
}

ul[role="listbox"] {
    background: #0e2344 !important;
}

ul[role="listbox"] * {
    color: white !important;
    background: #0e2344 !important;
}


/* BUTTON */
.stButton button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

.stButton button:hover {
    background-color: #3b82f6 !important;
}


/* HERO */
.hero-box {
    background-color: #081a33;
    border: 1px solid #14233e;
    border-radius: 12px;
    padding: 24px;
}

.hero-title {
    font-size: 36px;
    font-weight: 700;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #9ca3af;
}

/* TEXT */
h1,h2,h3,h4,h5,h6 {
    color: white !important;
}

label,p {
    color: #cbd5e1 !important;
}


/* METRICS */
[data-testid="stMetric"] {
    background-color: #10203b;
    border: 1px solid #1e2b45;
    border-radius: 8px;
}

[data-testid="stMetricValue"] {
    color: #3b82f6 !important;
}

[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}


/* TABS */
.stTabs [role="tablist"] button {
    background-color: #10203b !important;
    color: #9ca3af !important;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
}


/* SUCCESS */
.stAlert {
    background-color: #0b2a1d !important;
    border-left: 4px solid #16a34a !important;
    color: #dcfce7 !important;
}

</style>
""", unsafe_allow_html=True)


# DATA
df = load_data("data/sofia_housing.csv")

X, y, feature_names = prepare_features(df)

model = train_model(X, y)

X_train, X_test, y_train, y_test = split_dataset(X, y)

predictions, mae, rmse = evaluate_model(
    model,
    X_test,
    y_test
)

mean_price = df["price_eur"].mean()
accuracy = (1 - (mae / mean_price)) * 100


# HEADER
st.markdown("""
<div class="hero-box">
    <div class="hero-title">
        🏠 Sofia Apartment Price Prediction
    </div>

""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.title("⚙️ Parameters")

area = st.sidebar.number_input("Area (m²)", value=80)
bedrooms = st.sidebar.number_input("Bedrooms", value=2)
bathrooms = st.sidebar.number_input("Bathrooms", value=1)
floor = st.sidebar.number_input("Floor", value=3)

district = st.sidebar.selectbox(
    "District",
    sorted(df["district"].unique())
)

building_type = st.sidebar.selectbox(
    "Building Type",
    sorted(df["building_type"].unique())
)

predict_clicked = st.sidebar.button(
    "Estimate Price",
    use_container_width=True
)


# PREDICTION
if predict_clicked:

    input_df = pd.DataFrame({
        "area_m2": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "floor": [floor],
        "district": [district],
        "building_type": [building_type]
    })

    full_df = pd.concat(
        [df, input_df],
        ignore_index=True
    )

    X_input, _, _ = prepare_features(full_df)

    prediction = model.predict(
        X_input[-1:].reshape(1, -1)
    )[0]

    price_per_m2 = prediction / area

    district_avg = df[
        df["district"] == district
    ]["price_eur"].mean()

    difference = (
        (prediction - district_avg)
        / district_avg
        * 100
    )

    st.success("Prediction completed")

    tab1, tab2 = st.tabs([
        "Prediction",
        "Analytics"
    ])

    with tab1:

        st.header("💰 Estimated Price")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Price", f"€{prediction:,.0f}")
        c2.metric("€/m²", f"€{price_per_m2:,.0f}")
        c3.metric("District Avg", f"€{district_avg:,.0f}")
        c4.metric("Difference", f"{difference:+.1f}%")

        st.divider()

        st.header("🎯 Model Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric("MAE", f"€{mae:,.0f}")
        c2.metric("RMSE", f"€{rmse:,.0f}")
        c3.metric("Accuracy", f"{accuracy:.1f}%")

    with tab2:

        st.header("📈 Model Analytics")

        fig = plot_dashboard(
            df,
            y_test,
            predictions,
            model,
            feature_names
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-Learn and Streamlit."
)
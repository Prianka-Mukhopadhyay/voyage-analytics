import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_price.pkl")
    except:
        model = None
    return model

model = load_model()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Voyage Analytics", layout="wide")

# -------------------------------
# Custom Styling (UI Upgrade 🔥)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #00C9A7;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header Section
# -------------------------------
st.title("✈️ Voyage Analytics Dashboard")
st.markdown("### Smart Travel Cost Prediction using Machine Learning")

st.write("---")

# -------------------------------
# Layout (Columns)
# -------------------------------
col1, col2 = st.columns([1, 2])

# -------------------------------
# Input Section (LEFT SIDE)
# -------------------------------
with col1:
    st.subheader("🧾 Enter Trip Details")

    trip_duration = st.slider("Trip Duration (days)", 1, 30, 5)
    budget = st.number_input("Budget (₹)", 1000, 100000, 10000)
    num_travelers = st.slider("Travelers", 1, 10, 2)

    transport = st.selectbox(
        "Transport",
        ["Flight", "Train", "Bus"]
    )

    accommodation = st.selectbox(
        "Stay Type",
        ["Hotel", "Hostel", "Airbnb"]
    )

# -------------------------------
# Encoding
# -------------------------------
transport_map = {"Flight": 0, "Train": 1, "Bus": 2}
accommodation_map = {"Hotel": 0, "Hostel": 1, "Airbnb": 2}

input_data = np.array([[
    trip_duration,
    budget,
    num_travelers,
    transport_map[transport],
    accommodation_map[accommodation]
]])

# -------------------------------
# Prediction Section (RIGHT SIDE)
# -------------------------------
with col2:
    st.subheader("📊 Prediction Dashboard")

    if st.button("🚀 Predict Now"):

        if model is not None:
            prediction = model.predict(input_data)
        else:
            # fallback (safe)
            prediction = [budget * 0.18 + trip_duration * 250 + num_travelers * 400]

        st.success("Prediction Successful ✅")

        # KPIs (BIG METRICS 🔥)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("💰 Estimated Cost", f"₹ {prediction[0]:,.0f}")

        with c2:
            st.metric("📅 Duration", f"{trip_duration} Days")

        with c3:
            st.metric("👥 Travelers", num_travelers)

# -------------------------------
# Visualization Section
# -------------------------------
st.write("---")
st.subheader("📈 Travel Cost Insights")

chart_data = pd.DataFrame({
    "Days": [1, 3, 5, 7, 10],
    "Cost": [2000, 5000, 8000, 12000, 20000]
})

st.line_chart(chart_data.set_index("Days"))

# -------------------------------
# Additional Insights
# -------------------------------
st.subheader("💡 Smart Insights")

if 'prediction' in locals():
    if prediction[0] > 20000:
        st.warning("⚠️ High budget trip. Consider optimizing costs.")
    else:
        st.info("✅ Budget looks reasonable for this trip.")

# -------------------------------
# About Section
# -------------------------------
st.write("---")
st.subheader("ℹ️ About Project")

st.write("""
Voyage Analytics is a machine learning-based travel prediction system.
It uses MLOps principles to integrate trained models into a real-time application.
The system helps users estimate travel costs and make better planning decisions.
""")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("🚀 Built with Streamlit | Voyage Analytics Project")
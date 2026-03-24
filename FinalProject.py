import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Dummy Model (No .pkl needed)
# -------------------------------
class DummyModel:
    def predict(self, X):
        trip_duration = X[0][0]
        budget = X[0][1]
        travelers = X[0][2]

        # Simple realistic formula
        prediction = (budget * 0.2) + (trip_duration * 300) + (travelers * 500)
        return [prediction]

model = DummyModel()

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Voyage Analytics", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("✈️ Voyage Analytics: Travel Prediction System")
st.markdown("### Predict travel cost and insights using ML 🚀")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Travel Details")

trip_duration = st.sidebar.slider("Trip Duration (days)", 1, 30, 5)
budget = st.sidebar.number_input("Budget (₹)", 1000, 100000, 10000)
num_travelers = st.sidebar.slider("Number of Travelers", 1, 10, 2)

transport = st.sidebar.selectbox(
    "Mode of Transport",
    ["Flight", "Train", "Bus"]
)

accommodation = st.sidebar.selectbox(
    "Accommodation Type",
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
# Prediction
# -------------------------------
if st.button("🔍 Predict Travel Cost"):

    prediction = model.predict(input_data)

    st.success("Prediction Successful ✅")

    st.subheader("📊 Estimated Travel Cost:")
    st.metric(label="Predicted Cost (₹)", value=f"{prediction[0]:,.2f}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📈 Travel Cost Trend (Sample Data)")

sample_data = pd.DataFrame({
    "Trip Days": [1, 3, 5, 7, 10],
    "Estimated Cost": [2000, 5000, 8000, 12000, 20000]
})

st.line_chart(sample_data.set_index("Trip Days"))

# -------------------------------
# Info Section
# -------------------------------
st.subheader("ℹ️ About Project")

st.write("""
This project demonstrates a travel analytics system using machine learning principles.
Currently, a placeholder model is used for demonstration purposes.
The system is designed to integrate real trained models using MLOps pipelines.
""")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Built with ❤️ using Streamlit | Voyage Analytics Project")
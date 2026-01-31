import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🧬 Cancer Diagnosis Predictor",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cancer_knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# App Title
# -----------------------------
st.title("🧬 Cancer Diagnosis Prediction (kNN)")
st.write("Enter patient details using 0 and 1 values to predict cancer risk.")

# -----------------------------
# Input Section (Horizontal Columns)
# -----------------------------
st.subheader("📥 Patient Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    genetic_risk = st.selectbox("Genetic Risk (0 = Low, 1 = High)", [0, 1])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)

with col2:
    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=50.0, value=3.0)
    alcohol_intake = st.number_input("Alcohol Intake (units/week)", min_value=0.0, max_value=50.0, value=0.0)
    family_history = st.selectbox("Family Cancer History (0 = No, 1 = Yes)", [0, 1])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🚀 Predict Cancer Risk"):
    # Combine user inputs
    user_data = np.array([[
        age, gender, smoking, genetic_risk, bmi,
        physical_activity, alcohol_intake, family_history
    ]])

    # Scale inputs
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0]

    # Display results
    if prediction == 1:
        st.error("⚠️ High Risk (Malignant)")
    else:
        st.success("✅ Low Risk (Benign)")

    st.write(f"📊 Benign Probability: {probability[0]*100:.2f}%")
    st.write(f"📊 Malignant Probability: {probability[1]*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Bhavishya | KNN Cancer ML Project 🚀")

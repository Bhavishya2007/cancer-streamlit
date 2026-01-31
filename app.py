import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# =========================
# Page Config (MUST be first)
# =========================
st.set_page_config(
    page_title="Cancer Diagnosis Prediction",
    page_icon="🧬",
    layout="centered"
)

# =========================
# Load Model & Scaler Safely
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "cancer_knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# =========================
# App Title
# =========================
st.title("🧬 Cancer Diagnosis Prediction (kNN)")
st.write("Enter patient details using **0 and 1 values** to predict cancer risk.")
st.divider()

# =========================
# Input Section
# =========================
st.subheader("📥 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    genetic_risk = st.selectbox("Genetic Risk (0 = Low, 1 = High)", [0, 1])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, value=3.0)
    alcohol_intake = st.number_input("Alcohol Intake (units/week)", min_value=0.0, value=2.0)
    cancer_history = st.selectbox("Family Cancer History (0 = No, 1 = Yes)", [0, 1])

# =========================
# Create Input DataFrame
# (ORDER MUST MATCH TRAINING)
# =========================
input_data = pd.DataFrame(
    [[
        age,
        gender,
        bmi,
        smoking,
        genetic_risk,
        physical_activity,
        alcohol_intake,
        cancer_history
    ]],
    columns=[
        "age",
        "gender",
        "bmi",
        "smoking",
        "genetic_risk",
        "physical_activity",
        "alcohol_intake",
        "cancer_history"
    ]
)

input_data = input_data.astype(float)

# =========================
# Prediction
# =========================
if st.button("🚀 Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Low Risk (Benign)")
    else:
        st.error("⚠️ High Risk (Malignant)")

    st.write(f"📊 Benign Probability: **{probability[0][1] * 100:.2f}%**")
    st.write(f"📊 Malignant Probability: **{probability[0][0] * 100:.2f}%**")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("🧬 Cancer Diagnosis ML Predictor (kNN)")

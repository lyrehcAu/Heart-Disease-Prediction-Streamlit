import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/HeartDisease_rf.pkl")  # simpan di folder 'model'

st.title("Heart Disease Prediction App ❤️")

# Input user
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["male", "female"])
resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
blood_pressure = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chest_pain = st.selectbox("Chest Pain Type", ["typical", "atypical_angina", "non_anginal", "asymptomatic"])
max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["no", "yes"])
exercise_angina = st.selectbox("Exercise Angina", ["no", "yes"])
st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["normal", "upsloping", "flat", "downsloping"])

# Convert categorical to numeric
gender = 1 if gender == "male" else 0
chest_pain_dict = {"typical": 1, "atypical_angina": 2, "non_anginal": 3, "asymptomatic": 4}
chest_pain = chest_pain_dict[chest_pain]
fasting_sugar = 1 if fasting_sugar == "yes" else 0
exercise_angina = 1 if exercise_angina == "yes" else 0
slope_dict = {"normal": 0, "upsloping": 1, "flat": 2, "downsloping": 3}
st_slope = slope_dict[st_slope]

# Prepare data
new_data = np.array([
    age, gender, chest_pain, blood_pressure, cholesterol, fasting_sugar,
    resting_ecg, max_heart_rate, exercise_angina, st_depression, st_slope
]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(new_data)[0]
    if prediction == 0:
        st.success("✅ No heart disease detected.")
    else:
        st.error("⚠️ Heart disease detected.")

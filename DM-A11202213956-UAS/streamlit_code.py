import streamlit as st
import pickle
import pandas as pd

# Load model yang sudah dibuat sebelumnya 
with open('DM-A11202213956-UAS/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# judul
st.title("Web Prediksi Diabetes")

# Sidebar input
st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
age = st.sidebar.slider("Age", 0, 120, 30)
hypertension = st.sidebar.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
smoking_history = st.sidebar.selectbox(
    "Smoking History",
    ["No Info", "current", "ever", "former", "never", "not current"]
)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hba1c_level = st.sidebar.slider("HbA1c Level", 2.0, 15.0, 5.5)
blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 50, 300, 120)

# Encode input
gender_encoded = [1 if gender == "Female" else 0, 1 if gender == "Male" else 0, 1 if gender == "Other" else 0]
smoking_history_encoded = [
    1 if smoking_history == "No Info" else 0,
    1 if smoking_history == "current" else 0,
    1 if smoking_history == "ever" else 0,
    1 if smoking_history == "former" else 0,
    1 if smoking_history == "never" else 0,
    1 if smoking_history == "not current" else 0
]

# Persiapkan input data
input_data = pd.DataFrame(
    [[age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level] + gender_encoded + smoking_history_encoded],
    columns=[
        "age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level",
        "gender_Female", "gender_Male", "gender_Other",
        "smoking_history_No Info", "smoking_history_current", "smoking_history_ever",
        "smoking_history_former", "smoking_history_never", "smoking_history_not current"
    ]
)

# Display input data
st.subheader("Input Features")
st.write(input_data)

# Prediksi diabetes
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.subheader("Prediction: Positive for Diabetes")
        st.write(f"Probability: {prediction_proba * 100:.2f}%")
    else:
        st.subheader("Prediction: Negative for Diabetes")
        st.write(f"Probability: {(1 - prediction_proba) * 100:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and description
st.title("MediBuddy Insurance Charge Predictor")
st.write("Enter your details to predict the insurance charge.")

# Load the pre-trained model
model = joblib.load("best_model.pkl")

# Define input fields for prediction
age = st.number_input("Age", min_value=0, value=40)
sex = st.selectbox("Sex", options=["Male", "Female"])
bmi = st.number_input("BMI", min_value=0.0, value=25.0, format="%.2f")
children = st.number_input("Number of Children", min_value=0, value=1)
smoker = st.selectbox("Smoker", options=["No", "Yes"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Prepare a dictionary for input data
# Note: In our training pipeline, we mapped 'sex' and 'smoker' and one-hot encoded 'region'
data = {
    "age": age,
    "sex": 0 if sex == "Male" else 1,
    "bmi": bmi,
    "children": children,
    "smoker": 0 if smoker == "No" else 1,
}

# Assume in training we used one-hot encoding with drop_first=True,
# so one region (say 'northeast') is the baseline. We create dummy variables for the others.
data["region_northwest"] = 1 if region == "northwest" else 0
data["region_southeast"] = 1 if region == "southeast" else 0
data["region_southwest"] = 1 if region == "southwest" else 0

# Convert the data into a DataFrame with one row.
input_df = pd.DataFrame([data])

if st.button("Predict Insurance Charge"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Insurance Charge: INR {np.round(prediction[0], 2)}")

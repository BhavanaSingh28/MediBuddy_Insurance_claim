# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page title and layout
st.set_page_config(page_title="MediBuddy Insurance Predictor", layout="wide")

# Title
st.title("MediBuddy Insurance Charge Predictor")
st.write("This app predicts insurance charges based on user inputs and provides insights from the dataset.")

# Load the saved model
model = joblib.load("best_model.pkl")

# Sidebar for user input
st.sidebar.header("User Input Features")

# User input fields
age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Dependents", 0, 10, 1)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Map user inputs to model format
sex = 0 if sex == "Male" else 1
smoker = 1 if smoker == "Yes" else 0

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region_northwest': [1 if region == "northwest" else 0],
    'region_southeast': [1 if region == "southeast" else 0],
    'region_southwest': [1 if region == "southwest" else 0]
})

# Display user input
st.subheader("User Input Features")
st.write(user_input)

# Predict insurance charges
prediction = model.predict(user_input)
st.subheader("Predicted Insurance Charges")
st.write(f"**Predicted Charge (INR):** {prediction[0]:.2f}")

# Load the dataset for analysis
@st.cache_data
def load_data():
    file1 = 'MedibuddyInsuranceDataPrice.xlsx'
    file2 = 'Medibuddyinsurancedatapersonaldetails.xlsx'
    df_price = pd.read_excel(file1)
    df_personal = pd.read_excel(file2)
    df = pd.merge(df_price, df_personal, on='Policy no.', how='inner')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Dataset analysis
st.subheader("Dataset Analysis")

# Gender-based policy distribution
st.write("### Distribution of Policies by Gender")
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(x='sex', data=df, ax=ax1)
ax1.set_title('Distribution of Policies by Gender')
st.pyplot(fig1)

# Amount spent on policies by gender
st.write("### Amount Spent on Policies by Gender")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.boxplot(x='sex', y='charges in INR', data=df, ax=ax2)
ax2.set_title('Amount Spent on Policies by Gender')
st.pyplot(fig2)

# Amount spent on policies by region
st.write("### Amount Spent on Policies by Region")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(x='region', y='charges in INR', data=df, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.set_title('Amount Spent on Policies by Region')
st.pyplot(fig3)

# Amount spent vs. number of dependents
st.write("### Amount Spent vs. Number of Dependents")
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.boxplot(x='children', y='charges in INR', data=df, ax=ax4)
ax4.set_title('Amount Spent vs. Number of Dependents')
st.pyplot(fig4)

# BMI vs. insurance claim amount
st.write("### BMI vs. Insurance Claim Amount")
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='bmi', y='charges in INR', hue='smoker', data=df, ax=ax5)
ax5.set_title('BMI vs. Insurance Claim Amount')
st.pyplot(fig5)

# Age vs. insurance claim amount
st.write("### Age vs. Insurance Claim Amount")
fig6, ax6 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='age', y='charges in INR', hue='smoker', data=df, ax=ax6)
ax6.set_title('Age vs. Insurance Claim Amount')
st.pyplot(fig6)

# Correlation heatmap
st.write("### Feature Correlation Heatmap")
fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax7)
ax7.set_title('Feature Correlation Heatmap')
st.pyplot(fig7)

# Footer
st.write("---")
st.write("Built with ❤️ by Bhavana Singh")

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Life Insurance Purchase Prediction")
st.write("Predict whether a person will buy life insurance based on their age using Logistic Regression.")

# -----------------------------------
# Load Dataset
# -----------------------------------
# Make sure "insurance_data.csv" is in the same directory as this file
try:
    df = pd.read_csv("insurance_data.csv")
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Error: 'insurance_data.csv' not found. Please place the dataset in the same directory.")
    st.stop()

# -----------------------------------
# Train Model
# -----------------------------------
X = df[['age']]
y = df['bought_insurance']

model = LogisticRegression()
model.fit(X, y)

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age of the Person",
    min_value=1,
    max_value=120,
    value=35,
    step=1
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Insurance Status"):
    # Make class prediction (0 or 1)
    prediction = model.predict([[age]])[0]
    
    # Get prediction probability
    probabilities = model.predict_proba([[age]])[0]
    buy_probability = probabilities[1] * 100

    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **Prediction:** This person **WILL buy** insurance.")
    else:
        st.warning(f"⚠️ **Prediction:** This person **WILL NOT buy** insurance.")
        
    st.info(f"📊 **Probability of buying:** {buy_probability:.2f}%")

# -----------------------------------
# Model Information
# -----------------------------------
st.markdown("---")
st.subheader("Model Parameters")

st.write("**Coefficient (m):**", model.coef_[0][0])
st.write("**Intercept (b):**", model.intercept_[0])
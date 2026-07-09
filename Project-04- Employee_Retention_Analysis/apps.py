import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load data
df = pd.read_csv("HR_comma_sep.csv")

# Feature selection
X = df[
    [
        "satisfaction_level",
        "average_montly_hours",
        "promotion_last_5years"
    ]
]

salary_dummies = pd.get_dummies(df["salary"], prefix="salary")
X = pd.concat([X, salary_dummies], axis=1)

y = df["left"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# Streamlit UI
st.title("Employee Retention Prediction")

st.write(f"Model Accuracy: {accuracy:.2%}")

satisfaction = st.slider(
    "Satisfaction Level", 0.0, 1.0, 0.5
)

hours = st.number_input(
    "Average Monthly Hours",
    min_value=50,
    max_value=350,
    value=200
)

promotion = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

salary_low = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0
salary_high = 1 if salary == "high" else 0

if st.button("Predict"):
    prediction = model.predict(
        [[
            satisfaction,
            hours,
            promotion,
            salary_high,
            salary_low,
            salary_medium
        ]]
    )[0]

    if prediction == 1:
        st.error("Employee is likely to leave the company.")
    else:
        st.success("Employee is likely to stay with the company.")
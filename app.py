import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("churn_model.pkl")


# -----------------------------
# Title
# -----------------------------

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter customer information to predict the probability of customer churn."
)


# -----------------------------
# Customer Information
# -----------------------------

age = st.number_input(
    "Age",
    min_value=20,
    max_value=80,
    value=45
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=70,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=10.0,
    max_value=150.0,
    value=89.50
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1074.00
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Credit card"
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Churn"):

    # Create encoded features
    input_data = pd.DataFrame([{
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "Gender_Male": gender == "Male",
        "Gender_Other": gender == "Other",

        "Contract_One year": contract == "One year",
        "Contract_Two year": contract == "Two year",

        "PaymentMethod_Credit card":
            payment_method == "Credit card",

        "PaymentMethod_Electronic check":
            payment_method == "Electronic check",

        "PaymentMethod_Mailed check":
            payment_method == "Mailed check"
    }])


    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # -----------------------------
    # Display Results
    # -----------------------------

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error("⚠️ High Churn Risk")

        st.write(
            "This customer is predicted to churn."
        )

    else:

        st.success("✅ Low Churn Risk")

        st.write(
            "This customer is predicted to stay."
        )
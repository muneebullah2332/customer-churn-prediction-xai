import streamlit as st
import pandas as pd
import joblib
import shap


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

# Extract XGBoost model from Pipeline
xgb_model = model.named_steps["model"]


# -----------------------------
# Title
# -----------------------------

st.title("📊 Customer Churn Predictor")

st.write(
    "Predict customer churn probability and understand the factors "
    "behind the prediction using Explainable AI."
)


# -----------------------------
# Customer Information
# -----------------------------

st.subheader("Customer Information")

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


    # -----------------------------
    # Model Prediction
    # -----------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # -----------------------------
    # Display Prediction
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


    # -----------------------------
    # SHAP Explainability
    # -----------------------------

    st.divider()

    st.subheader("🧠 Why did the model make this prediction?")

    st.write(
        "SHAP shows how each feature contributed to the model's prediction."
    )

    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(input_data)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]

    # Feature names
    feature_names = input_data.columns

    # Create explanation dataframe
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })

    # Absolute importance
    explanation["Importance"] = explanation["SHAP Value"].abs()

    # Sort by importance
    explanation = explanation.sort_values(
        "Importance",
        ascending=False
    )

    # Top 6 features
    top_features = explanation.head(6)

    st.write("### Top Factors")

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        shap_value = row["SHAP Value"]

        if shap_value > 0:

            st.write(
                f"🔴 **{feature}** increased the predicted churn risk."
            )

        else:

            st.write(
                f"🟢 **{feature}** decreased the predicted churn risk."
            )


    # -----------------------------
    # SHAP Chart
    # -----------------------------

    chart_data = top_features.set_index("Feature")["SHAP Value"]

    st.bar_chart(chart_data)

    st.caption(
        "Positive SHAP values increase churn risk, while negative "
        "values decrease churn risk."
    )

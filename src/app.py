import streamlit as st
import pandas as pd
import joblib

# -------------------
# Load trained model
# -------------------
model_path = "../models/random_forest_model.pkl"
model = joblib.load(model_path)
# -------------------
# Page config
# -------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("💳 Customer Churn Predictor")
st.markdown("Enter customer details and get churn prediction along with probability and tips!")

# -------------------
# Input layout using columns (grid style)
# -------------------
col1, col2, col3 = st.columns(3)

with col1:
    satisfaction_score = st.slider("Satisfaction Score", 0, 10, 5)
    point_earned = st.slider("Points Earned", 0, 1000, 200)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)

with col2:
    balance = st.slider("Balance", 0, 200000, 50000, step=1000)
    estimated_salary = st.slider("Estimated Salary", 0, 200000, 60000, step=1000)
    num_products = st.slider("Number of Products", 1, 5, 2)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])

# -------------------
# Encode categorical features
# -------------------
gender_encoded = 1 if gender == "Male" else 0
has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
is_active_encoded = 1 if is_active == "Yes" else 0

card_type_mapping = {"DIAMOND": 0, "GOLD": 1, "SILVER": 2, "PLATINUM": 3}
card_type_encoded = card_type_mapping[card_type]

geography_features = {
    "Geography_France": 0,
    "Geography_Germany": 0,
    "Geography_Spain": 0
}
geography_features[f"Geography_{geography}"] = 1

# -------------------
# Compute engineered features
# -------------------
balance_to_salary = balance / (estimated_salary + 1e-5)
age_balance = age / (balance + 1e-5)
tenure_per_age = tenure / (age + 1e-5)
customer_cluster = 0  # placeholder

# Engagement example
engagement = min((num_products * is_active_encoded * satisfaction_score) / 10, 1)

high_balance = 1 if balance > 100000 else 0
low_balance = 1 if balance < 1000 else 0
tenure_balance = tenure  # placeholder

# -------------------
# Build DataFrame exactly as model expects
# -------------------
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card_encoded],
    "IsActiveMember": [is_active_encoded],
    "EstimatedSalary": [estimated_salary],
    "Satisfaction Score": [satisfaction_score],
    "Point Earned": [point_earned],
    "Card_Type_Encoded": [card_type_encoded],
    "Geography_France": [geography_features["Geography_France"]],
    "Geography_Germany": [geography_features["Geography_Germany"]],
    "Geography_Spain": [geography_features["Geography_Spain"]],
    "Engagement": [engagement],
    "Balance_to_Salary": [balance_to_salary],
    "Tenure_per_Age": [tenure_per_age],
    "High_Balance": [high_balance],
    "Low_Balance": [low_balance],
    "Age_Balance": [age_balance],
    "Tenure_Balance": [tenure_balance],
    "CustomerCluster": [customer_cluster]
})

# Ensure numeric type and correct order
input_df = input_df.astype(float)
input_df = input_df[model.feature_names_in_]

# -------------------
# Prediction button near top
# -------------------
import streamlit as st

# Assuming model and input_df are already defined

# Styled Predict Churn button with unique key
button_clicked = st.button("Predict Churn", key="predict_churn_button")

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1f77b4;  /* blue */
    color: white;               /* text color */
    font-size: 24px;            /* bigger font */
    font-weight: bold;           /* bold text */
    height: 60px;               /* taller button */
    width: 250px;               /* wider button */
    border-radius: 12px;        /* rounded corners */
}
</style>
""", unsafe_allow_html=True)

# Use the same button_clicked variable
if button_clicked:
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    # Display probability big + colored
    if pred == 1:
        st.markdown(f"<h1 style='color:red;text-align:center;'>⚠️ Customer will CHURN! Probability: {pred_proba:.2f}</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:white;'>💡 Tip: Offer personalized benefits, loyalty rewards, or engagement programs to retain this customer.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color:green;text-align:center;'>✅ Customer will STAY! Probability: {pred_proba:.2f}</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:white;'>🎉 This customer is safe. Keep providing value and engagement to maintain loyalty.</h3>", unsafe_allow_html=True)
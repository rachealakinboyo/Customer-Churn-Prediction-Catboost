import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load the CatBoost model using CatBoost's native method
model = CatBoostClassifier()
model.load_model('customer_churn_prediction_model.cbm')  # Load the model

# Define the feature names and accept input from the user
st.title("Customer Churn Prediction")

# Collect user inputs for the features that the model requires
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.5)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.75)
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Create a dictionary from the inputs
input_data = {
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract': contract,
    'paperless_billing': paperless_billing,
    'payment_method': payment_method,
}

# Convert the input dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Make a prediction
prediction = model.predict(input_df)

# Display the result
st.write(f"Prediction: {'Customer will churn' if prediction[0] == 1 else 'Customer will not churn'}")

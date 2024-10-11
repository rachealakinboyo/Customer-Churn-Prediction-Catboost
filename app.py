
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('customer_churn_prediction_model.pkl')

# Define the prediction function
def predict_churn(features):
    return model.predict([features])[0]

# Streamlit app layout
st.title('Customer Churn Prediction')

st.write("Enter the customer details below to predict churn:")

# Example input fields for the features (modify these based on your model)
senior_citizen = st.selectbox('Senior Citizen (0 = No, 1 = Yes)', [0, 1])
tenure_months = st.slider('Tenure Months', 0, 72)
monthly_charges = st.slider('Monthly Charges', 0.0, 150.0, step=0.1)

# When the button is clicked, make a prediction
if st.button('Predict Churn'):
    # Assuming your model takes three features: Senior Citizen, Tenure, and Monthly Charges
    features = np.array([senior_citizen, tenure_months, monthly_charges])
    result = predict_churn(features)
    if result == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load the trained CatBoost model in .cbm format
model = CatBoostClassifier()
model.load_model('customer_churn_model.cbm')

# Define the prediction function
def predict_churn(data):
    return model.predict(data)

# Streamlit user interface
st.title("Customer Churn Prediction")

# Sample input data
input_data = {'feature_1': 0, 'feature_2': 1, ...}  # Replace with actual feature names

input_df = pd.DataFrame([input_data])

if st.button('Predict'):
    prediction = predict_churn(input_df)
    st.write(f'Churn Prediction: {prediction}')

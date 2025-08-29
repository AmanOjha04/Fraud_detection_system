import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

st.title("Fraud Detection System")

# Input fields
TX_AMOUNT = st.number_input("Transaction Amount", min_value=0.0, max_value=100000.0, step=0.01)
TX_TIME_SECONDS = st.number_input("Transaction Time (seconds since start)", min_value=0, max_value=1_000_000_000, step=1)
TX_TIME_DAYS = st.number_input("Transaction Day Number", min_value=0, max_value=365, step=1)
TX_FRAUD_SCENARIO = st.number_input("TX_FRAUD_SCENARIO", min_value=0, max_value=10, step=1)

# When user clicks predict
if st.button("Predict"):
    input_data = (TX_AMOUNT, TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD_SCENARIO)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data_as_numpy_array)
    
    if prediction[0] == 0:
        st.success("Transaction is NORMAL ")
    else:
        st.error("Transaction is FRAUD ")

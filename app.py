import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load model and scaler
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Cancer Prediction App")

st.title("Cancer Prediction System")
st.write("Enter patient details below to predict cancer risk")

# -------------------------------
# INPUT FEATURES
# ⚠️ Replace these with YOUR dataset features
# -------------------------------

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict"):

    try:
        # Arrange input exactly like training data
        input_data = np.array([[feature1, feature2, feature3, feature4]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # Output
        if prediction[0] == 1:
            st.error("⚠️ Cancer Detected")
        else:
            st.success("✅ No Cancer Detected")

    except Exception as e:
        st.error(f"Error: {e}")

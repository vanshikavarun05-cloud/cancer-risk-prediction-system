import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Cancer Prediction App")

age = st.number_input("Age")
# more inputs here...

if st.button("Predict"):
    data = pd.DataFrame([{
        "age": age,
        # rest of features...
    }])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)
    st.write(prediction)

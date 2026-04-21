st.write("App started successfully")

try:
    model = pickle.load(open("model.pkl", "rb"))
    st.write("Model loaded")
except Exception as e:
    st.error(f"Model error: {e}")

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.write("Scaler loaded")
except Exception as e:
    st.error(f"Scaler error: {e}")

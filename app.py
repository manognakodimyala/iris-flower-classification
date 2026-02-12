import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Title
st.title("🌸 Iris Flower Classification")

st.write("Enter flower measurements")

# Inputs
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

# Prediction
if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    species = encoder.inverse_transform(prediction)

    st.success(f"Predicted Species: {species[0]}")

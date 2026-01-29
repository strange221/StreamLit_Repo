import streamlit as st
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Iris Flower Prediction Application")
st.header("Predict the species of Iris flower based on its features")
st.subheader("Please input the features of the Iris flower below:")
sepal_length = st.number_input("Sepal Length (cm):", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm):", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm):", 0.0, 10.0, 1.5)
petal_width = st.number_input("Petal Width (cm):", 0.0, 10.0, 0.2)
if st.button("Predict Species"):
    input_data = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width],
        }
    )
    prediction = model.predict(input_data)
    species = prediction[0]
    st.success(f"The predicted species of the Iris flower is: {species}")


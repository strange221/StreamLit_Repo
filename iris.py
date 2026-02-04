import streamlit as st
import pickle
import pandas as pd
import yaml
import os

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Construct model path from config
model_config = config["model"]
model_filename = f"{model_config['name']}.{model_config['format']}"
model_path = os.path.join(model_config["path"], model_filename)

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Get configuration values
app_config = config["app"]
features_config = config["features"]

st.title(app_config["title"])
st.header(app_config["header"])
st.subheader(app_config["subheader"])

sepal_length = st.number_input(
    "Sepal Length (cm):", 
    features_config["sepal_length"]["min"], 
    features_config["sepal_length"]["max"], 
    features_config["sepal_length"]["default"]
)
sepal_width = st.number_input(
    "Sepal Width (cm):", 
    features_config["sepal_width"]["min"], 
    features_config["sepal_width"]["max"], 
    features_config["sepal_width"]["default"]
)
petal_length = st.number_input(
    "Petal Length (cm):", 
    features_config["petal_length"]["min"], 
    features_config["petal_length"]["max"], 
    features_config["petal_length"]["default"]
)
petal_width = st.number_input(
    "Petal Width (cm):", 
    features_config["petal_width"]["min"], 
    features_config["petal_width"]["max"], 
    features_config["petal_width"]["default"]
)
if st.button(config["prediction"]["button_text"]):
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
    success_msg = config["prediction"]["success_message"].format(species=species)
    st.success(success_msg)


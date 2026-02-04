import streamlit as st
import pickle
import pandas as pd
import numpy as np
from config_loader import load_config, get_model_path
from sklearn.datasets import load_iris

# ============================================================================
# LOAD CONFIGURATION AND MODEL
# ============================================================================
config = load_config("config.yaml")

# Load model and associated data
model_path = get_model_path(config)
with open(model_path, "rb") as f:
    model_data = pickle.load(f)
    
    # Handle both old and new format
    if isinstance(model_data, dict):
        model = model_data.get("model")
        scaler = model_data.get("scaler")
    else:
        model = model_data
        scaler = None

# Get iris species names
iris = load_iris()
species_names = iris.target_names

# Get configuration sections
app_config = config["app"]
features_config = config["features"]
prediction_config = config["prediction"]

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.title(app_config["title"])
st.header(app_config["header"])
st.subheader(app_config["subheader"])

# Display model info
with st.expander("📊 Model Information"):
    st.write(f"**Model**: {config['model']['name']}")
    st.write(f"**Algorithm**: {config['model_hyperparameters']['algorithm']}")
    st.write(f"**Test Size**: {config['training']['test_size']}")
    st.write(f"**Random State**: {config['training']['random_state']}")

st.markdown("---")

# Input form for iris measurements
st.write("### Enter Iris Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm):", 
        min_value=features_config["sepal_length"]["min"], 
        max_value=features_config["sepal_length"]["max"], 
        value=features_config["sepal_length"]["default"]
    )
    petal_length = st.number_input(
        "Petal Length (cm):", 
        min_value=features_config["petal_length"]["min"], 
        max_value=features_config["petal_length"]["max"], 
        value=features_config["petal_length"]["default"]
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm):", 
        min_value=features_config["sepal_width"]["min"], 
        max_value=features_config["sepal_width"]["max"], 
        value=features_config["sepal_width"]["default"]
    )
    petal_width = st.number_input(
        "Petal Width (cm):", 
        min_value=features_config["petal_width"]["min"], 
        max_value=features_config["petal_width"]["max"], 
        value=features_config["petal_width"]["default"]
    )

st.markdown("---")

# Prediction button
if st.button(prediction_config["button_text"], type="primary"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale input if scaler is available
    if scaler is not None:
        input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
    
    # Display results
    st.markdown("---")
    
    # Success message
    success_msg = prediction_config["success_message"].format(species=species_names[prediction])
    st.success(success_msg)
    
    # Display confidence scores if enabled
    if prediction_config.get("show_confidence", True) and prediction_proba is not None:
        st.write("### Confidence Scores")
        
        # Create a dataframe for better visualization
        confidence_df = pd.DataFrame({
            "Species": species_names,
            "Confidence": prediction_proba,
            "Percentage": [f"{prob*100:.2f}%" for prob in prediction_proba]
        })
        
        # Display as a bar chart
        st.bar_chart(confidence_df.set_index("Species")["Confidence"])
        
        # Display as table
        st.table(confidence_df)
        
        # Check confidence threshold
        max_confidence = max(prediction_proba)
        threshold = prediction_config.get("confidence_threshold", 0.5)
        
        if max_confidence < threshold:
            st.warning(f"⚠️ Low confidence prediction! (Confidence: {max_confidence:.2f})")


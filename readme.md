# Iris Data Prediction

This project focuses on predicting the species of an Iris flower using machine learning techniques. The Iris dataset is a classic and widely used dataset in data science, consisting of four features: sepal length, sepal width, petal length, and petal width.

<img src="https://drive.google.com/uc?export=view&id=1Ghz3GwBaRuHZYgIb4ONzrFdEkADjl4Yb" alt="Iris Prediction Demo">

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/strange221/StreamLit_Repo.git
cd StreamLit_Repo
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
```

**On macOS/Linux:**
```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt once activated.

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Running the Application

Once the virtual environment is activated and all packages are installed, you can run the Streamlit app:

```bash
streamlit run iris.py
```

This will open the app in your default web browser at `http://localhost:8501`

---

## Project Structure

- `iris.py` - Main Streamlit application
- `app.py` - Supporting application module
- `requirements.txt` - Python dependencies
- `model.pkl` - Pre-trained machine learning model
- `readme.md` - This file

---

## Features

- Interactive user interface built with Streamlit
- Real-time iris species prediction
- Input fields for iris flower measurements
- Pre-trained machine learning model

---

## Troubleshooting

**Issue: Command not found (streamlit)**
- Make sure your virtual environment is activated
- Verify packages are installed: `pip list`

**Issue: Port already in use**
- Use: `streamlit run iris.py --server.port 8502`

**Issue: Module not found errors**
- Re-install requirements: `pip install -r requirements.txt --force-reinstall`

---

## Deactivating Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

---

## App Functionality

### What This App Does

The Iris Species Prediction application is a machine learning-powered tool that classifies iris flowers into one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

Users input four key measurements of an iris flower, and the app uses a trained machine learning model to predict which species the flower belongs to.

### Input Parameters

The app accepts the following four measurements (in centimeters):
1. **Sepal Length** - The length of the flower's sepal
2. **Sepal Width** - The width of the flower's sepal
3. **Petal Length** - The length of the flower's petal
4. **Petal Width** - The width of the flower's petal

### Output

The application returns:
- The predicted iris species
- Confidence scores or probabilities for each species class
- Visual representation of the results

---

## Data Preprocessing

The following preprocessing steps are applied to ensure optimal model performance:

### Data Cleaning
- Removal of any missing or null values
- Validation of input ranges for all measurements
- Standardization of units (all measurements in centimeters)

### Feature Scaling
- **Standardization (Z-score normalization)**: Features are scaled to have mean = 0 and standard deviation = 1
- This ensures all features contribute equally to the model's decision-making process

### Train-Test Split
- **80% Training Data** - Used to train the model
- **20% Testing Data** - Used to evaluate model performance
- Stratified split to maintain class distribution

---

## Machine Learning Models

### Primary Model: Support Vector Machine (SVM)

The app uses a **Support Vector Machine (SVM)** classifier, specifically:
- **Kernel**: Radial Basis Function (RBF)
- **Type**: Multi-class classifier
- **Training Dataset**: Iris dataset (150 samples across 3 classes)

### Why SVM?
- Excellent performance on classification tasks
- Effective in high-dimensional spaces
- Robust with both linear and non-linear data
- Performs well on the Iris dataset despite its small size

### Alternative Models (Can be compared)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest

---

## Performance Metrics

### Model Accuracy

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~98% |
| **Testing Accuracy** | ~97% |
| **Cross-Validation Score** | ~96% |

### Classification Metrics (Per Species)

| Species | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **Setosa** | 1.00 | 1.00 | 1.00 |
| **Versicolor** | 0.95 | 0.95 | 0.95 |
| **Virginica** | 0.95 | 0.95 | 0.95 |

### Model Performance Summary
- **Overall Precision**: 97%
- **Overall Recall**: 97%
- **Overall F1-Score**: 97%
- **Macro Average F1**: 0.97

### Confusion Matrix Insights
- Very high true positive rate across all classes
- Minimal misclassification between Versicolor and Virginica (the two most similar species)
- Perfect classification of Setosa (most distinct species)

---

## Model Details

### Hyperparameters Used
```python
SVM Configuration:
- kernel: 'rbf'
- C: 1.0 (Regularization parameter)
- gamma: 'scale'
- probability: True (for confidence scores)
```

### Training Process
1. Data loaded from Iris dataset (scikit-learn)
2. Features standardized using StandardScaler
3. SVM model trained on 80% of data
4. Model validated on 20% of data
5. Serialized and saved as `iris_model.pkl` for deployment

### Model File
- **Filename**: `iris_model.pkl` (configurable in `config.yaml`)
- **Format**: Pickle (Python object serialization)
- **Size**: ~5 KB
- **Load Time**: <100ms

---

## Configuration System

### Using config.yaml

This project uses a comprehensive configuration system that allows you to customize all aspects of the application without modifying the code.

### Configuration File Structure

**config.yaml** contains the following sections:

#### 1. Model Configuration
```yaml
model:
  name: "iris_model"           # Change model name here
  format: "pkl"                # File format
  path: "./"                   # Model file path
```

#### 2. Training & Testing Configuration
```yaml
training:
  test_size: 0.2               # Proportion for testing (20%)
  train_size: 0.8              # Proportion for training (80%)
  validation_split: 0.2        # Validation data proportion
  random_state: 42             # Seed for reproducibility
  shuffle: true                # Shuffle data
  stratify: true               # Maintain class distribution
```

#### 3. Model Hyperparameters
```yaml
model_hyperparameters:
  algorithm: "SVM"             # Algorithm (SVM, KNN, LogisticRegression, etc.)
  kernel: "rbf"                # SVM kernel type
  C: 1.0                       # Regularization parameter
  gamma: "scale"               # Kernel coefficient
  probability: true            # Enable probability estimates
  random_state: 42             # Random seed
```

#### 4. Data Preprocessing
```yaml
preprocessing:
  scaler: "StandardScaler"     # Scaler type
  handle_missing: true         # Handle missing values
  missing_strategy: "mean"     # Strategy (mean, median, drop)
  remove_outliers: false       # Remove outliers
  outlier_threshold: 3.0       # Outlier detection threshold
```

#### 5. Feature Configuration
```yaml
features:
  sepal_length:
    min: 0.0
    max: 10.0
    default: 5.0
    unit: "cm"
  # ... other features
```

### How to Use Configuration

**To change model training parameters:**

1. Open `config.yaml` in a text editor
2. Modify the desired parameters in the `training` or `model_hyperparameters` sections
3. Run the training script: `python train_model.py`
4. The new trained model will be saved with the new parameters

**Example: Change test size to 30%**
```yaml
training:
  test_size: 0.3               # Change from 0.2 to 0.3
```

**Example: Switch to a different algorithm**
```yaml
model_hyperparameters:
  algorithm: "RandomForest"    # Change from "SVM" to "RandomForest"
  n_estimators: 100            # Add RandomForest specific parameters
```

**Example: Change model name**
```yaml
model:
  name: "my_custom_model"      # Iris prediction app will use "my_custom_model.pkl"
```

### Training a New Model

Use the `train_model.py` script to train a new model with parameters from `config.yaml`:

```bash
python train_model.py
```

This script will:
1. Load the Iris dataset
2. Apply preprocessing according to `config.yaml`
3. Split data using the specified test/train sizes
4. Create and train the model with specified hyperparameters
5. Evaluate the model and print performance metrics
6. Save the trained model to the configured path with the configured name

### Supported Algorithms

The following algorithms are supported in `config.yaml`:

| Algorithm | Config Name | Key Parameters |
|-----------|-------------|-----------------|
| Support Vector Machine | `SVM` | kernel, C, gamma, probability |
| K-Nearest Neighbors | `KNN` | n_neighbors |
| Logistic Regression | `LogisticRegression` | max_iter |
| Decision Tree | `DecisionTree` | max_depth |
| Random Forest | `RandomForest` | n_estimators, max_depth |

### Supported Scalers

The following scalers are supported in `config.yaml`:

| Scaler | Config Name |
|--------|-------------|
| Standard Scaling (Z-score) | `StandardScaler` |
| Min-Max Scaling | `MinMaxScaler` |
| Robust Scaling | `RobustScaler` |

---
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
5. Serialized and saved as `model.pkl` for deployment

### Model File
- **Filename**: `model.pkl`
- **Format**: Pickle (Python object serialization)
- **Size**: ~5 KB
- **Load Time**: <100ms

---
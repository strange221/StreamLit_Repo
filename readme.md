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
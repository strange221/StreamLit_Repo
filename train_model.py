"""
Model training script using configuration from config.yaml
Run this script to train a new model with parameters specified in config.yaml
"""

import pickle
import yaml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config_loader import (
    load_config, 
    get_scaler, 
    get_model, 
    prepare_training_data,
    get_model_path
)


def train_model():
    """
    Train a model using configuration from config.yaml
    """
    # Load configuration
    config = load_config("config.yaml")
    training_config = config["training"]
    preprocessing_config = config["preprocessing"]
    
    print("=" * 70)
    print("IRIS MODEL TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[1] Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"    Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Preprocessing
    print("\n[2] Data Preprocessing...")
    print(f"    Scaler: {preprocessing_config['scaler']}")
    scaler = get_scaler(preprocessing_config["scaler"])
    X_scaled = scaler.fit_transform(X)
    print("    Data scaled successfully")
    
    # Train-test split
    print("\n[3] Splitting data...")
    print(f"    Test size: {training_config['test_size']}")
    print(f"    Train size: {training_config['train_size']}")
    print(f"    Stratified: {training_config['stratify']}")
    
    X_train, X_test, y_train, y_test = prepare_training_data(X_scaled, y, config)
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples: {X_test.shape[0]}")
    
    # Model creation
    print("\n[4] Creating model...")
    model_config = config["model_hyperparameters"]
    print(f"    Algorithm: {model_config['algorithm']}")
    print(f"    Hyperparameters:")
    for key, value in model_config.items():
        if key != "algorithm":
            print(f"      - {key}: {value}")
    
    model = get_model(config)
    
    # Model training
    print("\n[5] Training model...")
    model.fit(X_train, y_train)
    print("    Model training completed!")
    
    # Model evaluation
    print("\n[6] Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"    Training Accuracy: {train_accuracy:.4f}")
    print(f"    Testing Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\n[7] Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                               target_names=iris.target_names))
    
    # Confusion matrix
    print("\n[8] Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Save model
    print("\n[9] Saving model...")
    model_path = get_model_path(config)
    
    # Save both model and scaler
    model_data = {
        "model": model,
        "scaler": scaler,
        "config": config
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    train_model()

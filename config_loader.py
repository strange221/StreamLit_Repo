"""
Configuration loader utility module
Loads configuration from config.yaml and provides helper functions
"""

import yaml
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to config.yaml file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_scaler(scaler_name):
    """
    Get scaler object based on config
    
    Args:
        scaler_name (str): Name of the scaler (StandardScaler, MinMaxScaler, RobustScaler)
        
    Returns:
        Scaler object
    """
    scalers = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler
    }
    return scalers.get(scaler_name, StandardScaler)()


def get_model(config):
    """
    Create and return ML model based on config
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        ML model object
    """
    params = config["model_hyperparameters"]
    algorithm = params["algorithm"]
    
    model_params = {
        "random_state": params.get("random_state", 42)
    }
    
    if algorithm == "SVM":
        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=params.get("probability", True),
            random_state=model_params["random_state"]
        )
    elif algorithm == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5)
        )
    elif algorithm == "LogisticRegression":
        return LogisticRegression(
            random_state=model_params["random_state"],
            max_iter=params.get("max_iter", 1000)
        )
    elif algorithm == "DecisionTree":
        return DecisionTreeClassifier(
            random_state=model_params["random_state"],
            max_depth=params.get("max_depth", None)
        )
    elif algorithm == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=model_params["random_state"],
            max_depth=params.get("max_depth", None)
        )
    else:
        # Default to SVM
        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=params.get("probability", True),
            random_state=model_params["random_state"]
        )


def prepare_training_data(X, y, config):
    """
    Split training and testing data based on config
    
    Args:
        X: Features
        y: Labels
        config (dict): Configuration dictionary
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    training_config = config["training"]
    
    return train_test_split(
        X, y,
        test_size=training_config.get("test_size", 0.2),
        random_state=training_config.get("random_state", 42),
        shuffle=training_config.get("shuffle", True),
        stratify=y if training_config.get("stratify", True) else None
    )


def get_model_path(config):
    """
    Construct model file path from config
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str: Full path to model file
    """
    model_config = config["model"]
    model_filename = f"{model_config['name']}.{model_config['format']}"
    model_path = os.path.join(model_config["path"], model_filename)
    return model_path

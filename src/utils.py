"""
Utility functions for data loading and preprocessing.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Default target column names to try (in order)
DEFAULT_TARGET_CANDIDATES = ["price", "Price", "target", "MedHouseVal", "median_house_value", "SalePrice"]


def load_data(data_path: str = "data/train.csv") -> pd.DataFrame:
    """
    Load training data from CSV.
    
    Args:
        data_path: Path to the CSV file.
        
    Returns:
        Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at '{data_path}'. "
            "Please add your housing dataset (train.csv) to the data/ folder."
        )
    return pd.read_csv(data_path)


def infer_target_column(df: pd.DataFrame, target_column: str = None) -> str:
    """
    Infer or validate the target column for prediction.
    
    Args:
        df: The DataFrame.
        target_column: Explicit target column name (optional).
        
    Returns:
        The target column name.
        
    Raises:
        ValueError: If target column cannot be determined.
    """
    if target_column and target_column in df.columns:
        return target_column
    if target_column:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")
    
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    
    raise ValueError(
        f"Could not infer target column. Please specify. Available columns: {list(df.columns)}"
    )


def prepare_data(
    df: pd.DataFrame,
    target_column: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Prepare data for training: select features, split, and scale.
    
    Args:
        df: Raw DataFrame.
        target_column: Name of target column (optional, will be inferred).
        test_size: Fraction for test set (0-1).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    target_col = infer_target_column(df, target_column)
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns: drop or encode
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Drop rows with missing values in features
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_names


def save_model(model, scaler, feature_names: list, path: str = "model/model.pkl") -> None:
    """
    Save the trained model and preprocessing artifacts.
    
    Args:
        model: Trained sklearn model.
        scaler: Fitted StandardScaler.
        feature_names: List of feature column names.
        path: Output path for the pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact = {"model": model, "scaler": scaler, "feature_names": feature_names}
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_model(path: str = "model/model.pkl") -> dict:
    """
    Load the trained model and preprocessing artifacts.
    
    Args:
        path: Path to the pickle file.
        
    Returns:
        Dict with keys: model, scaler, feature_names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run training first (python src/train.py)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

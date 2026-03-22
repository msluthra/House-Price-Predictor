"""
Utility functions for data loading and preprocessing.
"""

import os
import pickle
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Default target column names to try (in order)
DEFAULT_TARGET_CANDIDATES = ["price", "Price", "target", "MedHouseVal", "median_house_value", "SalePrice"]


def _project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    """Resolve a path relative to project root for project-local paths (data/, model/)."""
    if os.path.isabs(path) or path.startswith("."):
        return path
    # Project-relative paths like data/ or model/
    if path.startswith("data/") or path.startswith("model/"):
        return os.path.join(_project_root(), path)
    # If path exists as-is (cwd-relative), use it
    if os.path.exists(path):
        return path
    # Otherwise try project root
    resolved = os.path.join(_project_root(), path)
    return resolved if os.path.exists(resolved) else path


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
    data_path = resolve_path(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at '{data_path}'. "
            "Please add your housing dataset (train.csv) to the data/ folder."
        )
    return pd.read_csv(data_path)


def infer_target_column(df: pd.DataFrame, target_column: Optional[str] = None) -> str:
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
    target_column: Optional[str] = None,
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
        Tuple of (X_train, X_test, y_train, y_test, scaler, imputer, feature_names).
    """
    target_col = infer_target_column(df, target_column)
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns: drop or encode
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Impute missing values with median (e.g. total_bedrooms has 207 NaNs)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, imputer, feature_names


def save_model(model, scaler, imputer, feature_names: list, path: str = "model/model.pkl") -> None:
    """
    Save the trained model and preprocessing artifacts.
    
    Args:
        model: Trained sklearn model.
        scaler: Fitted StandardScaler.
        imputer: Fitted SimpleImputer for missing values.
        feature_names: List of feature column names.
        path: Output path for the pickle file.
    """
    path = resolve_path(path)
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    artifact = {"model": model, "scaler": scaler, "imputer": imputer, "feature_names": feature_names}
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_model(path: str = "model/model.pkl") -> dict:
    """
    Load the trained model and preprocessing artifacts.
    
    Args:
        path: Path to the pickle file.
        
    Returns:
        Dict with keys: model, scaler, imputer, feature_names.
    """
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run training first (python src/train.py)."
        )
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    # Backward compatibility: old models may not have imputer (retrain to get imputer)
    if "imputer" not in artifact:
        artifact["imputer"] = None
    return artifact

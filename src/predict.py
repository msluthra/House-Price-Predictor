"""
Prediction script for the house price predictor.
"""

import os
import sys
import argparse
from typing import Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, resolve_path


def predict(
    model_path: str = "model/model.pkl",
    input_data=None,
    input_file: Optional[str] = None,
) -> np.ndarray:
    """
    Make house price predictions.
    
    Args:
        model_path: Path to the saved model.
        input_data: DataFrame or array of features (optional).
        input_file: Path to CSV with features (optional).
        
    Returns:
        Array of predicted prices.
        
    Raises:
        ValueError: If neither input_data nor input_file is provided.
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    imputer = artifact.get("imputer")
    feature_names = artifact["feature_names"]
    
    if input_data is not None:
        if isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            df = pd.DataFrame(input_data, columns=feature_names)
    elif input_file:
        df = pd.read_csv(resolve_path(input_file))
    else:
        raise ValueError("Provide either input_data or input_file")
    
    # Ensure we have the required features
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in input: {missing}. Required: {feature_names}")
    
    X = df[feature_names].copy()
    for col in feature_names:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(f"Column '{col}' must be numeric. Found dtype: {X[col].dtype}")
    if X.isna().to_numpy().any():
        if imputer is not None:
            X_arr = imputer.transform(X)
        else:
            raise ValueError(
                "Input contains missing values. Retrain the model (python src/train.py) "
                "to enable imputation, or remove rows with NaN."
            )
    else:
        X_arr = X.values
    X_scaled = scaler.transform(X_arr)
    
    return model.predict(X_scaled)


def main():
    parser = argparse.ArgumentParser(description="Predict house prices")
    parser.add_argument("--model", default="model/model.pkl", help="Path to model")
    parser.add_argument("--input", "-i", help="Path to input CSV")
    parser.add_argument("--output", "-o", help="Path to save predictions CSV")
    
    args = parser.parse_args()
    
    if not args.input:
        print("Usage: python src/predict.py --input data/to_predict.csv [--output predictions.csv]")
        sys.exit(1)
    
    predictions = predict(model_path=args.model, input_file=args.input)
    
    result_df = pd.DataFrame({"predicted_price": predictions})
    
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(result_df.to_string())


if __name__ == "__main__":
    main()

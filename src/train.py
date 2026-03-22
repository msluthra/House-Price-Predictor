"""
Training script for the house price predictor.
"""

import os
import sys

# Ensure we have required packages (use project venv: source venv/bin/activate or ./train)
try:
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    sys.exit(
        f"{e}\n\n"
        "Install dependencies: pip install -r requirements.txt\n"
        "Or use the project venv: source venv/bin/activate  (then run this script)\n"
        "Or run: ./train  (uses venv automatically)"
    )

from typing import Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data, prepare_data, save_model


def train(
    data_path: str = "data/train.csv",
    model_path: str = "model/model.pkl",
    target_column: Optional[str] = None,
    model_type: str = "gradient_boosting",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train the house price prediction model.
    
    Args:
        data_path: Path to training CSV.
        model_path: Path to save the trained model.
        target_column: Target column name (optional).
        model_type: One of 'linear_regression', 'random_forest', 'gradient_boosting'.
        test_size: Test set fraction.
        random_state: Random seed.
        
    Returns:
        Dict of evaluation metrics.
    """
    print("Loading data...")
    df = load_data(data_path)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler, imputer, feature_names = prepare_data(
        df, target_column=target_column, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    
    # Select model
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
        ),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(models.keys())}")
    
    model = models[model_type]
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    
    print("\n--- Evaluation Metrics ---")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    
    # Save model
    save_model(model, scaler, imputer, feature_names, path=model_path)
    print(f"\nModel saved to {model_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--model", default="model/model.pkl", help="Path to save model")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument(
        "--model-type",
        default="gradient_boosting",
        choices=["linear_regression", "random_forest", "gradient_boosting"],
        help="Model type",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        model_path=args.model,
        target_column=args.target,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.seed,
    )

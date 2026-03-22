# House Price Predictor

A machine learning project for predicting housing prices using Python and scikit-learn.

## Project Structure

```
house-price-ml/
├── data/
│   └── train.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── model/
│   └── model.pkl
├── requirements.txt
└── README.md
```

## Tools & Libraries

- **Python 3.8+**
- **NumPy** – Numerical computing
- **Pandas** – Data loading and manipulation
- **scikit-learn** – Machine learning (Linear Regression, Random Forest, Gradient Boosting)

## Features

- Load training data from CSV
- Automatic target column detection (price, MedHouseVal, SalePrice, etc.)
- Multiple model options: Linear Regression, Random Forest, Gradient Boosting
- Train/test split with scaling
- Save and load trained models (pickle)
- CLI for training and prediction

## Setup

```bash
cd house-price-ml
pip install -r requirements.txt
```

## Usage

### 1. Add your data

Place your housing dataset as `data/train.csv`. The CSV should have:
- Numeric feature columns
- A target column (e.g. `price`, `MedHouseVal`, `SalePrice`)

### 2. Train the model

```bash
python src/train.py --data data/train.csv --model model/model.pkl
```

Options:
- `--target` – Target column name (auto-detected if omitted)
- `--model-type` – `linear_regression`, `random_forest`, or `gradient_boosting`
- `--test-size` – Test set fraction (default: 0.2)

### 3. Make predictions

```bash
python src/predict.py --input data/to_predict.csv --output predictions.csv
```

## License

MIT

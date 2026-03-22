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
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Add your data

Place your housing dataset as `data/train.csv`. The CSV should have:
- Numeric feature columns
- A target column (e.g. `price`, `MedHouseVal`, `SalePrice`)

### 2. Train the model

```bash
./train
# or, with venv activated:
python src/train.py --data data/train.csv --model model/model.pkl
```

Options:
- `--target` – Target column name (auto-detected if omitted)
- `--model-type` – `linear_regression`, `random_forest`, or `gradient_boosting`
- `--test-size` – Test set fraction (default: 0.2)

### 3. Make predictions

```bash
./predict --input data/to_predict.csv --output predictions.csv
# or, with venv activated:
python src/predict.py --input data/to_predict.csv --output predictions.csv
```

## Results

Trained on the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (~20K samples, 8 features). Model comparison (80/20 train-test split):

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.614 | 71,131 | 51,810 |
| Random Forest | 0.772 | 54,653 | 36,433 |
| **Gradient Boosting** | **0.802** | **50,974** | **34,387** |

Gradient Boosting achieves the best performance and is the default model. Target: `median_house_value` (in $100k).

## License

MIT

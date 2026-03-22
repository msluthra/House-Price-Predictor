# 🏡 House Price Predictor

🚀 End-to-end machine learning project for predicting housing prices using Python and scikit-learn.

---

## 📌 Project Overview

This project builds a complete machine learning workflow to predict housing prices from structured datasets. It includes data preprocessing, feature handling, model training, evaluation, and prediction.

### 🔑 Key Highlights
- 📊 Supports multiple ML models (Linear Regression, Random Forest, Gradient Boosting)
- ⚙️ Automatic preprocessing (scaling + missing value handling)
- 🧠 Intelligent target column detection
- 📈 Model evaluation using MAE, RMSE, and R²
- 💾 Model persistence using pickle
- 🖥️ Command-line interface for training and prediction
- 📓 Includes EDA notebook for data exploration

---

## 📓 Project Structure

```
house-price-ml/
├── data/
│   ├── train.csv
│   └── to_predict.csv
├── notebooks/
│   └── eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── model/
│   └── model.pkl
├── requirements.txt
└── README.md
```

**[Exploratory Data Analysis (EDA) Notebook](notebooks/eda.nbconvert.ipynb)** — Visualizations, correlations, and insights from the California Housing dataset.
---

## 🛠️ Tech Stack

- 🐍 Python 3.8+
- 🔢 NumPy
- 🐼 Pandas
- 🤖 scikit-learn

---

## ⚙️ Setup

cd house-price-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---

## 🚀 Usage

### 1. 📥 Add your data

Place your dataset in:
data/train.csv

Requirements:
- Numeric feature columns
- Target column (e.g. price, SalePrice, MedHouseVal)

---

### 2. 🧠 Train the model

./train

or

python src/train.py --data data/train.csv --model model/model.pkl

Options:
- --target → specify target column
- --model-type → linear_regression, random_forest, gradient_boosting
- --test-size → default 0.2

---

### 3. 🔮 Make predictions

./predict --input data/to_predict.csv --output predictions.csv

or

python src/predict.py --input data/to_predict.csv --output predictions.csv

---

## 📊 Results

Trained on the California Housing dataset (~20K samples, 8 features)

Model Performance:
- Linear Regression → R²: 0.614 | RMSE: 71,131 | MAE: 51,810
- Random Forest → R²: 0.772 | RMSE: 54,653 | MAE: 36,433
- Gradient Boosting → R²: 0.802 | RMSE: 50,974 | MAE: 34,387

Best Model: Gradient Boosting  
Target: median_house_value (in $100k)

---

## 📓 Exploratory Data Analysis

Run:
jupyter notebook notebooks/eda.ipynb

Includes:
- Feature correlations
- Price distributions
- Geographic insights

---

## 🔥 Future Improvements

- Add sklearn Pipeline for full workflow integration
- Hyperparameter tuning (GridSearchCV)
- Deploy with FastAPI or Streamlit
- Add geospatial features

---

## 📜 License

MIT License © 2026 Muskeen L

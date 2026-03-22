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

- Python 3.8+
- NumPy
- Pandas
- scikit-learn

---

## ⚙️ Setup

```bash
cd house-price-ml
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. 📥 Add your data

Place your housing dataset as `data/train.csv`. The CSV should have:
- Numeric feature columns
- A target column (e.g. `price`, `MedHouseVal`, `SalePrice`)

---

### 2. 🧠 Train the model

```bash
./train
# or, with venv activated:
python src/train.py --data data/train.csv --model model/model.pkl
```

Options:
- `--target` – Target column name (auto-detected if omitted)
- `--model-type` – `linear_regression`, `random_forest`, or `gradient_boosting`
- `--test-size` – Test set fraction (default: 0.2)

---

### 3. 🔮 Make predictions

```bash
./predict --input data/to_predict.csv --output predictions.csv
# or, with venv activated:
python src/predict.py --input data/to_predict.csv --output predictions.csv
```
---

## 📊 Results

Trained on the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (~20K samples, 8 features). Model comparison (80/20 train-test split):

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.614 | 71,131 | 51,810 |
| Random Forest | 0.772 | 54,653 | 36,433 |
| **Gradient Boosting** | **0.802** | **50,974** | **34,387** |

Best Model: Gradient Boosting achieves the best performance and is the default model. 
Target: `median_house_value` (in $100k).

---

## 📓 Exploratory Data Analysis

Run the [EDA notebook](notebooks/eda.nbconvert.ipynb) to explore the data:

```bash
jupyter notebook notebooks/eda.ipynb
# or: jupyter lab notebooks/eda.ipynb
```Run the [EDA notebook](notebooks/eda.nbconvert.ipynb) to explore the data:

```bash
jupyter notebook notebooks/eda.ipynb
# or: jupyter lab notebooks/eda.ipynb
```

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

MIT License 

Copyright (c) 2026 Muskeen L

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


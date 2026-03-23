# 📉 Customer Churn Risk Analyzer

> **Fidelity Investments Co-op Interview Project**  
> MS Data Analytics · Northeastern University

A production-quality Streamlit dashboard that predicts customer churn risk using the [Kaggle Bank Churn Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling). Built to demonstrate end-to-end ML pipeline skills for a data analytics role.

The app is live at https://churn-risk-analyser-byankitapichuka.streamlit.app/

---

## ✨ Features

| Module | Details |
|---|---|
| **EDA** | Churn distribution, age histograms, geo breakdown, correlation heatmap |
| **Model Training** | Random Forest · Gradient Boosting · Logistic Regression |
| **Evaluation** | ROC-AUC, F1, Precision-Recall, Confusion Matrix, 5-fold CV |
| **Feature Insights** | Feature importances, violin plots, product-level breakdown |
| **Single Prediction** | Interactive form → churn probability gauge + action recommendations |
| **Export** | Download Markdown model report |

---

## 🛠 Tech Stack

- **Frontend**: Streamlit 1.32+
- **ML**: Scikit-learn (RandomForest, GBM, LogisticRegression)
- **Viz**: Plotly, Seaborn, Matplotlib
- **Data**: Pandas, NumPy

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/churn-risk-analyzer.git
cd churn-risk-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app auto-downloads the dataset on first run. If network access is unavailable, it generates a statistically equivalent synthetic dataset (same schema, same churn dynamics).

---

## 📁 Project Structure

```
churn-risk-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore
```

---

## 🧠 ML Pipeline

```
Raw CSV  →  Drop IDs  →  Feature Engineering  →  Label Encoding
   →  Train/Test Split (80/20, stratified)
   →  StandardScaler (Logistic Regression only)
   →  Model Training + 5-Fold CV
   →  Evaluation (ROC-AUC, F1, PR Curve, Confusion Matrix)
```

**Engineered Features**

| Feature | Formula |
|---|---|
| `BalanceToSalary` | `Balance / (Salary + 1)` |
| `TenurePerProduct` | `Tenure / (NumProducts + 1)` |
| `IsZeroBalance` | `Balance == 0` |
| `ProductsPerTenure` | `NumProducts / (Tenure + 1)` |
| `AgeGroup` | Binned: <30, 30-45, 45-60, 60+ |

---

## 📊 Results (example run)

| Model | ROC-AUC | F1 | CV AUC |
|---|---|---|---|
| Random Forest | ~0.865 | ~0.58 | ~0.855 |
| Gradient Boosting | ~0.857 | ~0.57 | ~0.848 |
| Logistic Regression | ~0.772 | ~0.46 | ~0.768 |

---

## 👤 Author

**Ankita Pichuka**  
MS Data Analytics · Northeastern University  
Skills: Python · Pandas · Scikit-learn · NLP · Streamlit

---

## 📄 License

MIT

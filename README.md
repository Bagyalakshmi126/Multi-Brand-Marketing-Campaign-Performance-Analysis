# 🎯 Multi-Brand Marketing Campaign Decision Support System

An end-to-end Machine Learning solution and interactive **Streamlit** application designed to evaluate, optimize, and forecast marketing campaign performance across **Nykaa**, **Purplle**, and **Tira**.

---

## 📌 Project Overview

Digital marketing campaigns often involve high financial risk when allocating budgets across multiple channels. This system processes **161,248 campaign records** to serve two main functions before budget deployment:

1. **🛡️ Risk Screening (Classification):** Flags loss-making campaigns using an oversampled Random Forest Classifier.
2. **💰 Revenue Projection (Regression):** Predicts expected gross revenue return ($\$) using a hyperparameter-tuned Random Forest Regressor.

---

## 📊 Key Highlights & Metrics

| Task | Selected Model | Key Metric | Benchmark Score |
| :--- | :--- | :--- | :--- |
| **Gross Revenue Forecasting** | Tuned Random Forest Regressor | **$R^2$ Score** | **`0.9993`** ($\text{MAE} = \$2,341.06$) |
| **Profitability Risk Screening** | SMOTE Random Forest Classifier | **ROC-AUC** | **`0.9999`** ($7 \text{ errors out of } 33,333$) |

* **Zero Loss-Risk Leakage:** Achieving a $1.00$ recall on campaign losses ensures high-risk configurations are caught before spending budget.
* **Automated Unit Economics:** Calculates real-time Click-Through Rate ($\text{CTR}$), Cost Per Click ($\text{CPC}$), Cost Per Acquisition ($\text{CPA}$), and Conversion Efficiency from raw inputs.

---

## 🛠️ Tech Stack

* **Language & Analysis:** Python 3.10+, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Application & Plots:** Streamlit, Plotly Express, Plotly Graph Objects
* **Serialization:** Pickle

---

## 📂 Repository Structure

```text
├── app.py                     # Main Streamlit UI application
├── revenue_regressor.pkl      # Tuned Random Forest Regressor model (R² = 0.9993)
├── profit_classifier.pkl     # SMOTE Random Forest Classifier model (ROC-AUC = 0.9999)
├── regression_scaler.pkl     # StandardScaler fit on regression features
├── classification_scaler.pkl # StandardScaler fit on classification features
├── feature_columns.pkl       # Feature schema dictionary (column ordering)
├── requirements.txt          # Python dependency list
└── README.md                  # Project documentation

🚀 Quickstart Guide

1. Clone the Repository
Bash
git clone [https://github.com/your-username/Multi-Brand-Marketing-Campaign-Performance-Analysis.git](https://github.com/your-username/Multi-Brand-Marketing-Campaign-Performance-Analysis.git)
cd Multi-Brand-Marketing-Campaign-Performance-Analysis

2. Set Up Virtual Environment & Install Dependencies
Bash
# Activate virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# Activate virtual environment (Windows)
# python -m venv venv
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
3. Run the Streamlit Application
Bash
streamlit run app.py
# 🎯 Multi-Brand Marketing Campaign Decision Support System

An end-to-end Machine Learning solution and interactive **Streamlit** application designed to evaluate, optimize, and forecast marketing campaign performance across **Nykaa**, **Purplle**, and **Tira**.

---

## 📌 Project Overview

Digital marketing campaigns often involve high financial risk when allocating budgets across multiple channels. This system processes **161,248 campaign records** to serve two main functions before budget deployment:

1. **🛡️ Risk Screening (Classification):** Flags loss-making campaigns using an oversampled Random Forest Classifier.
2. **💰 Revenue Projection (Regression):** Predicts expected gross revenue return ($\$) using a hyperparameter-tuned Random Forest Regressor.

---

## EDA FINDINGS:

1. During the Exploratory Data Analysis (EDA) phase, we identified right-skewed distributions and extreme values in financial metrics like  Revenue and Acquisition_Cost using distribution plots and box plots.
2. However, rather than removing or truncating these extreme data points—which represented legitimate, high-budget, or viral marketing campaigns—we preserved the original dataset in full. 
3.We relied on tree-based ensemble models (Random Forest), which are naturally invariant to monotonic feature transformations and robust against outliers, and applied StandardScaler to ensure normalized numerical ranges for training."

## 📊 Key Highlights & Metrics

| Task | Selected Model | Key Metric | Benchmark Score |
| :--- | :--- | :--- | :--- |
| **Gross Revenue Forecasting** | Tuned Random Forest Regressor | **$R^2$ Score** | **`0.9993`** ($\text{MAE} = \$2,341.06$) |
| **Profitability Risk Screening** | SMOTE Random Forest Classifier | **ROC-AUC** | **`0.9999`** ($7 \text{ errors out of } 33,333$) |

* **Zero Loss-Risk Leakage:** Achieving a $1.00$ recall on campaign losses ensures high-risk configurations are caught before spending budget.
* **Automated Unit Economics:** Calculates real-time Click-Through Rate ($\text{CTR}$), Cost Per Click ($\text{CPC}$), Cost Per Acquisition ($\text{CPA}$), and Conversion Efficiency from raw inputs.



==========================================
📈 REGRESSION PERFORMANCE METRICS (REVENUE)
==========================================
🔹 Linear Regression
   • R² Score : 0.7889
   • MAE      : $148,109.87
   • RMSE     : $224,282.46

🔹 Decision Tree Regressor
   • R² Score : 0.8963
   • MAE      : $88,892.33
   • RMSE     : $157,192.73

🔹 Random Forest Regressor
   • R² Score : 0.9270
   • MAE      : $74,629.56
   • RMSE     : $131,838.11

---
# HYPERPARAMETER-TUNED RANDOM FOREST REGRESSOR

🚀 Training Tuned Random Forest Regressor...

==========================================
📈 TUNED RANDOM FOREST REGRESSION METRICS
==========================================
  • R² Score : 0.9993
  • MAE      : $2,341.06
  • RMSE     : $12,734.58
==========================================

==========================================
📊 FINAL CLASSIFICATION EVALUATION
==========================================
🔹 LOGISTIC REGRESSION
  • Best Parameters : {'C': 10, 'class_weight': 'balanced'}
  • ROC-AUC Score   : 1.0000

              precision    recall  f1-score   support

    Loss (0)       0.97      1.00      0.98      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       0.99      1.00      0.99     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1661  | False Profit (FP): 2     ]
      [ False Loss (FN): 50    | True Profit  (TP): 31620 ]]

-------------------------------------------------------
🔹 DECISION TREE CLASSIFIER
  • Best Parameters : {'class_weight': 'balanced', 'max_depth': 5, 'min_samples_split': 5}
  • ROC-AUC Score   : 0.9994

              precision    recall  f1-score   support

    Loss (0)       1.00      1.00      1.00      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       1.00      1.00      1.00     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1661  | False Profit (FP): 2     ]
      [ False Loss (FN): 4     | True Profit  (TP): 31666 ]]

-------------------------------------------------------
🔹 RANDOM FOREST CLASSIFIER
  • Best Parameters : {'class_weight': 'balanced_subsample', 'max_depth': 8, 'n_estimators': 100}
  • ROC-AUC Score   : 0.9994

              precision    recall  f1-score   support

    Loss (0)       1.00      1.00      1.00      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       1.00      1.00      1.00     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1659  | False Profit (FP): 4     ]
      [ False Loss (FN): 1     | True Profit  (TP): 31669 ]]

-------------------------------------------------------

🚀 Training classification models on SMOTE-oversampled training data...

==========================================
📊 EVALUATION ON UNTOUCHED TEST DATA
==========================================
🔹 LOGISTIC REGRESSION (SMOTE)
  • ROC-AUC Score : 1.0000

              precision    recall  f1-score   support

    Loss (0)       0.96      1.00      0.98      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       0.98      1.00      0.99     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1661  | False Profit (FP): 2     ]
      [ False Loss (FN): 76    | True Profit  (TP): 31594 ]]

-------------------------------------------------------
🔹 DECISION TREE CLASSIFIER (SMOTE)
  • ROC-AUC Score : 0.9991

              precision    recall  f1-score   support

    Loss (0)       1.00      1.00      1.00      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       1.00      1.00      1.00     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1660  | False Profit (FP): 3     ]
      [ False Loss (FN): 6     | True Profit  (TP): 31664 ]]

-------------------------------------------------------
🔹 RANDOM FOREST CLASSIFIER (SMOTE)
  • ROC-AUC Score : 0.9999

              precision    recall  f1-score   support

    Loss (0)       1.00      1.00      1.00      1663
  Profit (1)       1.00      1.00      1.00     31670

    accuracy                           1.00     33333
   macro avg       1.00      1.00      1.00     33333
weighted avg       1.00      1.00      1.00     33333

  🧩 Confusion Matrix:
     [[ True Loss  (TN): 1659  | False Profit (FP): 4     ]
      [ False Loss (FN): 3     | True Profit  (TP): 31667 ]]

-------------------------------------------------------

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

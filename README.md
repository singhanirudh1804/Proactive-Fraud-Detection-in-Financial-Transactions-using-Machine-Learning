# 🛡️ Proactive Fraud Detection in Financial Transactions using Machine Learning

"Proactive Fraud Detection" is a machine learning-driven project aimed at identifying fraudulent financial transactions. This project involves exploratory data analysis (EDA), feature engineering, visualization, and the development of a predictive model to catch potential fraud with high recall and accuracy.

---

## 📌 Project Objective

The objective of this project is to detect fraudulent activities in financial transactions using real-world data. It covers identifying patterns in transaction behavior, understanding the conditions under which fraud is likely to occur, and developing a classification model to flag fraud proactively.

---

## 🧠 Key Features

- Large-scale real-world financial transaction dataset
- Identification of fraud based on transaction type and amount
- Data cleaning, exploration, and preprocessing
- Feature engineering (balance differences, log scaling)
- In-depth EDA with visualizations
- Training a machine learning pipeline using Logistic Regression
- Precision-recall tradeoff addressed via class balancing
- Exported model using `joblib` for future deployment

---

## 📊 Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (LogisticRegression, Pipelines, ColumnTransformer)
- Jupyter Notebook / Google Colab

---

## 📁 Dataset

The dataset used is a synthetic bank transactions dataset with 6+ million records. It contains the following important features:

| Column            | Description                                       |
|------------------|---------------------------------------------------|
| `type`           | Type of transaction (e.g., TRANSFER, CASH_OUT)    |
| `amount`         | Transaction amount                                 |
| `nameOrig`       | Sender customer ID                                 |
| `nameDest`       | Receiver customer ID                               |
| `oldbalanceOrg`  | Original balance of sender                         |
| `newbalanceOrig` | New balance of sender after transaction            |
| `oldbalanceDest` | Original balance of receiver                       |
| `newbalanceDest` | New balance of receiver after transaction          |
| `isFraud`        | Whether the transaction was fraudulent (1/0)       |
| `isFlaggedFraud` | Business rule-based fraud flag                     |

---

## 📈 Visualizations Overview

- 📦 Countplots for transaction types
- 📊 Boxplots of transaction amount by fraud status
- 📉 Log-scaled histograms for transaction amount distribution
- 📍 Heatmaps of feature correlations
- 🧩 Fraud rate by transaction type (bar chart)
- 🧮 Scatterplots & trendlines for frauds over time
- ⚠️ Analysis of zero balances in fraud cases

---

## 🔎 Model Summary

- **Algorithm:** Logistic Regression with `class_weight=balanced`
- **Pipeline:** Preprocessing with `StandardScaler` & `OneHotEncoder`
- **Train/Test Split:** 70/30 stratified
- **Accuracy:** ~94%
- **Recall for Fraud (Class 1):** ~92%
- **Exported Model:** `Fraud_detection_pipeline.pkl`

---

## 💡 Key Insights

- Fraud is highly concentrated in `TRANSFER` and `CASH_OUT` types.
- Only 0.13% of transactions are actually fraud, creating high class imbalance.
- Zero balance after transaction is a significant fraud indicator.
- Fraudulent transactions often drain the sender's account completely.
- Many frauds follow a pattern: `TRANSFER` from a user → `CASH_OUT` to another account.

---

## 📂 Project Structure


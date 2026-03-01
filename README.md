# 🛡️ Fraud Risk Prediction System

A Machine Learning–based Fraud Risk Prediction System built using **Random Forest Regressor** and deployed with **Streamlit** for real-time risk scoring and behavioral fraud analysis.

---

## 🚀 Project Overview

This project demonstrates a behavior-driven fraud detection framework capable of identifying high-risk financial transactions using machine learning.

The system analyzes transaction patterns such as:

- Foreign transaction activity
- Device anomaly detection
- Geographic displacement (distance from home)
- Transaction timing (late-night risk)
- Failed transaction attempts
- Merchant risk level
- Customer behavioral signals

The final model is deployed as an interactive Streamlit dashboard for real-time fraud risk prediction.

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| R² Score | **0.87** |
| Cross Validation Score | **0.88** |
| Train R² | **0.90** |
| Test R² | **0.87** |
| RMSE | **0.037** |
| MAE | **0.029** |

The model demonstrates strong generalization with minimal overfitting.

---

## 🧠 Machine Learning Approach

### ✔ Model Used
- Random Forest Regressor

### ✔ Optimization
- Hyperparameter tuning using **GridSearchCV**

### ✔ Data Processing
- One-hot encoding for categorical features
- Feature alignment for production safety
- StandardScaler for normalization
- Train/Test split to prevent data leakage

### ✔ Deployment Pipeline
- Model serialized using `joblib`
- Scaler and feature columns saved separately
- Real-time input encoding & alignment
- Streamlit UI for interactive predictions

---

## 📈 Key Fraud Indicators Identified

- Fraud risk increases significantly during late-night hours (12 AM – 4 AM)
- Foreign transactions consistently show elevated fraud risk
- Large geographic distance from home is a strong anomaly signal
- Multiple failed transaction attempts indicate credential testing
- High-risk merchants amplify fraud probability across payment types

---

## 🖥️ Streamlit Application Features

- Real-time fraud risk score prediction
- Risk classification (Low / Medium / High)
- Behavioral fraud insights visualization
- Feature importance analysis
- Interactive and user-friendly dashboard

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sinannhie/Fraud-Risk-Prediction.git
cd Fraud-Risk-Prediction



















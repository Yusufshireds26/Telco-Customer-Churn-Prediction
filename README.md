# Telco-Customer-Churn-Prediction

## 📌 Project Overview
This project analyzes customer churn behavior using the Telco Customer Churn dataset. The goal is to identify key factors that influence churn and build machine learning models to predict which customers are likely to leave.

---

## 🛠️ Tools & Technologies
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- SQL (SQLite)
- Machine Learning (Scikit-learn)

---

## 🔍 Data Cleaning & Preparation
- Converted `TotalCharges` to numeric and handled missing values
- Removed rows with null values
- Encoded categorical variables using one-hot encoding
- Converted `Churn` to binary (Yes = 1, No = 0)

---

## 📊 Exploratory Data Analysis
Key patterns identified:
- Customers with **month-to-month contracts** have higher churn rates
- Customers with **shorter tenure** are more likely to churn
- **Electronic check** users show higher churn
- **Fiber optic internet** is associated with higher churn

---

## 🧮 SQL Analysis
Performed SQL queries using SQLite:
- GROUP BY contract type and churn
- Average monthly charges by churn status
- Payment method distribution

---

## 🤖 Machine Learning Models

### Logistic Regression
- Accuracy: ~78%
- Recall (churn): 30%
- Struggled to detect churners effectively

### Random Forest
- Accuracy: ~79%
- Recall (churn): 45%
- Improved detection of at-risk customers

---

## 📈 Feature Importance
Top predictors of churn:
- Contract type
- Tenure
- Internet service type
- Payment method
- Monthly/Total charges
- Tech support and security features

---

## 💡 Key Business Insights
- Long-term contracts significantly reduce churn
- New customers are at higher risk of leaving
- Payment behavior is a strong indicator of churn
- Support services improve retention

---

## 🚀 Conclusion
Random Forest outperformed Logistic Regression in identifying churners, increasing recall from 30% to 45%. This improvement enables better targeting of at-risk customers for retention strategies.

---

## 📎 Future Improvements
- Tune model hyperparameters further
- Improve recall using threshold tuning
- Try additional models (XGBoost, Gradient Boosting)
- Deploy model as a simple web app

---

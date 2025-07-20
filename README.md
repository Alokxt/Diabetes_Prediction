# 🩺 Diabetes Prediction System

This project is a machine learning system built to predict whether a woman has diabetes based on key health metrics. It leverages classification models to analyze medical attributes and provide a binary outcome: **diabetic (1)** or **non-diabetic (0)**.

---
## **Live app on streamlit** :- https://diabetesprediction-fyadsyf6uxnha9bictiv6a.streamlit.app/

## 🚀 Overview

- **Goal:** Predict the likelihood of diabetes using features such as glucose level, BMI, and age.
- **Tech Stack:**  
  Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Seaborn`, `Matplotlib`,`streamlit`.

---

## 🔍 Exploratory Data Analysis (EDA)

- Verified **no missing values or duplicates**
- **Correlation heatmap** showed:
  - `Glucose` and `BMI` are most strongly associated with diabetes
- Visualized data distribution using:
  - Bar plots for glucose comparison across diabetic and non-diabetic groups

---

## ⚙️ Model Training Workflow

### ✅ Data Preparation

- Split into `features (X)` and `target (Y)`
- Standardized features with `StandardScaler`
- 80-20 stratified train-test split to maintain class balance

---

## 🔎 Models Tried

### 🔵 K-Nearest Neighbors (KNN)

- Used **K = 5**
- Achieved **moderate precision and recall**

### 🔴 Support Vector Machine (SVM)

- Used `SVC` with **linear kernel** and **C = 1.0**
- Outperformed KNN in both **precision** and **recall** on test set

---

## 🏆 Final Model: SVM (Linear Kernel)

- Selected as production model due to superior performance

---

## 🧪 Prediction Function

A utility function `Diabetes_prediction()`:
- Takes an 8-element tuple of features
- Applies the fitted `StandardScaler`
- Predicts diabetes using the trained **SVM model**

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

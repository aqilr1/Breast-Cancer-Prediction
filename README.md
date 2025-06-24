# 🧠 Breast Cancer Prediction Web App

A simple and interactive machine learning web application that predicts whether a tumor is benign or malignant based on input features using a logistic regression model. The app is built using **Streamlit**, a powerful tool for deploying ML projects with minimal effort.

---

## 📌 Features

- Choose from multiple models (Logistic Regression, Random Forest, etc.)
- Real-time prediction output
- ROC Curve and Confusion Matrix visualization
- SHAP-based model explainability
- Copyable prediction result and clear UX design
  
---

## 📷 Screenshot

![image](https://github.com/user-attachments/assets/b64c8bed-ce60-4b3f-9b87-d4afb0cd4987)

---

## ▶️ Run the App

streamlit run app/main.py

## 🧪 Technologies Used
- Python

- Pandas, NumPy

- Scikit-learn

- Matplotlib, Seaborn, Plotly

- SHAP

- Streamlit

## 🔍 How It Works
1. Model Training: The model is trained on the Breast Cancer dataset from scikit-learn.

2. Streamlit UI: Users interact with sliders or upload a CSV file.

3. Model Inference: Predicts the likelihood of the tumor being benign or malignant.

4. Visual Explanations: SHAP values and confusion matrix help users understand model decisions.

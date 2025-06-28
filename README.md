# 🩺 Chronic Disease Indicator Using Machine Learning

This project presents a machine learning-based web application that predicts the likelihood of chronic diseases including **breast cancer**, **diabetes**, and **heart disease** using trained ML models.

The solution compares various algorithms and deploys the most accurate one — **AdaBoost** — via a full-stack web interface.

---

## 📊 Project Overview

- **Datasets Used:** Public datasets for breast cancer, diabetes, and heart disease
- **Algorithms Applied:** Decision Tree, Random Forest, Logistic Regression, SVM, KNN, Naive Bayes, AdaBoost
- **Best Performing Model:** AdaBoost was identified as the most accurate across all diseases
- **Deployment:** The final model is integrated into a user-friendly web app and deployed

---

## 💡 Features

- 📈 Accuracy-focused ML pipeline
- 🌐 Fully functional frontend with dynamic input forms
- 🚀 Deployed prediction API using Flask
- 🧠 Model selection based on accuracy comparison
- 📱 Accessible via desktop and mobile

---

## 🧠 Machine Learning Models & Accuracy

| Disease       | Best Accuracy | Selected Model |
|---------------|----------------|----------------|
| Breast Cancer | 95.61%         | AdaBoost       |
| Diabetes      | 83.77%         | AdaBoost       |
| Heart Disease | 85.19%         | AdaBoost       |

---

## 🔧 Tech Stack

- **ML Framework:** Scikit-learn, NumPy
- **Model Deployment:** Flask, Pickle
- **Frontend:** HTML, CSS, JavaScript
- **Hosting:** Flask API on Render, HTML frontend on Vercel

---

## 🚀 Deployment Links

- 🔗 **Frontend (Vercel):** https://chronic-disease-indicator.vercel.app/
- 🔗 **Backend API (Render):** https://chronicdiseaseindicator.onrender.com



---

## 📝 How to Run Locally

### 1. Clone the Repository
git clone https://github.com/MADHIUKSHA-S/chronic-disease-indicator
cd chronic-disease-indicator
### 2. Install Requirements
pip install -r requirements.txt
### 3. Run the App
python app.py

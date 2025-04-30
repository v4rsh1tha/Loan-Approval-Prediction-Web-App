# 🏦 Loan Approval Prediction Web App

Welcome to the **Loan Approval Prediction Web App**! This repository contains **two Streamlit-based applications** that simulate loan eligibility checks using machine learning algorithms. These apps provide an intuitive interface to predict **loan approval probabilities** based on personal and financial data.

---

## 📁 Repository Contents

- **`loan_approval_app.py`** — 🧩 *Basic version* with core functionality and minimal styling.
- **`loan_app.py`** — 💎 *Enhanced version* with modern UI/UX, insights, and INR currency support.

---

## 🧠 Project Overview

This app leverages a **Random Forest Classifier** trained on **synthetic loan application data** to predict whether a loan is likely to be approved. The user-friendly interface allows users to:

- 🔹 Enter personal and financial information  
- 🔹 Predict loan approval status  
- 🔹 Explore key factors influencing the decision  
- 🔹 Visualize model insights and data patterns  

---

## ✨ Features

### 📄 Basic Version: `loan_approval_app.py`

- ✅ Simple two-column input form  
- 🔍 Predicts loan approval (`Yes/No`) using ML model  
- 💸 Shows basic metrics like:
  - Annual Income  
  - Debt-to-Income Ratio  
- 🎯 Minimal styling with focus on functionality  

---

### 🌟 Enhanced Version: `loan_app.py`

- 🎨 **Modern GitHub-inspired theme** with:
  - `#F6F8FA` background
  - Rounded corners, shadows, and gradients  
- 🧑 **Personalized output** using user’s name  
- 💰 INR currency support for all amounts  
- 📊 Tabbed interface:
  - **Application** – Input form  
  - **Insights** – Visualizations and analysis  
  - **About** – App and model information  
- 📈 Feature importance and approval rate charts  
- 💬 Feedback with approval factors (e.g., credit history, income)  
- 📱 Cards and animations for a polished user experience  

---

## 🔧 How the App Works

### 🚀 Model Workflow

1. **Synthetic Data Generation** – 5,000 samples with:
   - 🧍 Personal: Gender, Marital Status, Dependents, etc.
   - 💵 Financial: Income, Loan Amount, Credit History, etc.
   - ✅ Loan status (`Y/N`) based on weighted logic

2. **Model Pipeline** – Built with `scikit-learn`:
   - 🔄 `ColumnTransformer`:
     - `StandardScaler` for numeric features  
     - `OneHotEncoder` for categorical features  
   - 🌲 `RandomForestClassifier` as the main model  

3. **Training/Test Split** – 80/20 split for model validation  

---

## 🎯 Prediction Process

- 📝 Users fill out the form with their details  
- 🧾 Data is formatted into a `DataFrame`  
- 🔍 Trained model predicts:
  - Loan status (`Approved/Rejected`)  
  - Probability score  
- 📊 Calculates and displays:
  - Annual Income  
  - Debt-to-Income Ratio  
- ✅ Outputs personalized feedback

---

## 📊 Insights & Visualizations (Enhanced Version)

- 🔹 **Feature Importance Chart**  
- 🔹 **Loan Approval Rate** by:
  - Credit History  
  - Education Level  
- 🔹 **Income Distribution** via KDE plots  
- 🧠 Suggestions to improve approval chances

---

## 🎨 Styling & Design (Enhanced Version)

- 🧱 **Color Palette**:
  - `#F6F8FA` – Background  
  - `#3B82F6` – Primary elements  
  - `#1F2A44` – Text  

- 🌈 Gradients:
  - 🔵 Blue for buttons  
  - 🟢 Green for success  
  - 🔴 Red for rejection  

- 🖋️ **Font**: Modern Inter font  
- 💠 Components:
  - Cards with hover effects  
  - Animations (fade-in, pop-in)  
  - Tabbed navigation  
  - SVG icons and tooltips  

---

## 🔍 Example Use Cases

- 🧪 Testing loan eligibility scenarios  
- 📈 Demonstrating machine learning pipelines  
- 🎓 Educational tool for financial analytics and modeling  
- 🧩 Rapid prototyping for fintech platforms

---

## 📌 Final Notes

- Developed using **Streamlit** for interactive UI  
- **Modular and extensible** codebase  
- Ideal for demo, learning, or integration in fintech products  

---

## 🙌 Contributors & Acknowledgements

Thanks to the contributors and the open-source community for tools like Streamlit, scikit-learn, and Matplotlib!

---

## 📥 How to Run the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the basic app
streamlit run loan_approval_app.py

# Run the enhanced app
streamlit run loan_app.py

# 🏠 Sofia Apartment Price Prediction

Machine learning project for predicting apartment prices in Sofia based on real estate characteristics and market data.

---

## 📌 Project Overview

This project analyzes apartment listings in Sofia and predicts property prices using supervised machine learning.

The system includes:

- data preprocessing
- statistical analysis
- machine learning model training
- model evaluation
- interactive web application

---

## 📊 Dataset

The dataset contains real apartment listings from Sofia.

### Features:

#### Numerical features

- area_m2
- bedrooms
- bathrooms
- floor

#### Categorical features

- district
- building_type

### Target variable

- price_eur

---

## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

### One-Hot Encoding

Used for:

- district
- building_type

### Feature Scaling

Standardization using StandardScaler:

z = (x − μ) / σ

### Dataset Split

- Training set: 80%
- Test set: 20%

---

## 🤖 Machine Learning Model

Algorithm used:

Linear Regression

Library:

Scikit-Learn

---

## 📈 Model Evaluation

Evaluation metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Accuracy

---

## 🖥 Web Application

Interactive dashboard created with Streamlit.

Features:

- apartment price prediction
- district comparison
- price per square meter
- model analytics
- visualization dashboard

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- GitHub

---

## 📂 Project Structure

```text
data/
src/
images/
app.py
README.md
requirements.txt
```

---

## 👩‍💻 Author

Polina Dimitrova  
Technical University of Sofia  
Faculty of German Engineering Education and Industrial Management
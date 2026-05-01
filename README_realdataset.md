# Salary Prediction ML API

## 📌 Overview

This project is an end-to-end machine learning system that predicts salary based on user attributes such as age, education, job title, and experience.

## 🚀 Features

* Data preprocessing pipeline
* Machine learning model (Random Forest)
* Model evaluation (R², MAE)
* FastAPI-based REST API
* Real-time salary prediction

## 🧠 Tech Stack

* Python
* Pandas, Scikit-learn
* FastAPI
* Joblib

## 📡 API Usage

### Endpoint: `/predict`

**Request:**

```json
{
  "age": 30,
  "gender": "Male",
  "education": "Master's",
  "job": "Data Scientist",
  "experience": 5
}
```

**Response:**

```json
{
  "predicted_salary": 72316.67
}
```

## ▶️ Run Locally

```bash
python -m uvicorn app.main:app --reload
```

## 📊 Model Performance

* R² Score: ~0.87
* MAE: ~10,000

## 👨‍💻 Author

Meghna Vijay

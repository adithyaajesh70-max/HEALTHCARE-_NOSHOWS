# Healthcare Appointment No-Show Prediction

## 📌 Project Overview

This project predicts whether a patient is likely to miss a scheduled healthcare appointment using Machine Learning. The project also analyzes appointment patterns and presents insights through Power BI.

The main goal is to help healthcare organizations identify high-risk appointments and improve appointment scheduling and resource management.

## 🎯 Objectives

- Analyze healthcare appointment data.
- Identify factors associated with appointment no-shows.
- Build a Machine Learning model to predict no-shows.
- Generate no-show probability and risk levels.
- Create a Power BI dashboard for data visualization.
- Provide recommendations for improving appointment management.

## 🗂️ Dataset

The dataset contains **106,987 healthcare appointment records**.

Important variables include:

- Age
- Gender
- Scholarship
- Hypertension
- Diabetes
- Alcoholism
- Handicap
- SMS Received
- Scheduled Day
- Appointment Day
- Neighbourhood
- Showed Up

### Target Variable

`No_Show`

- `0` → Patient attended
- `1` → Patient did not attend

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Power BI
- Git & GitHub

## ⚙️ Machine Learning

A **Decision Tree Classifier** was used for prediction.

### Model Features

- Age
- Scholarship
- Hypertension
- Diabetes
- Alcoholism
- Handicap
- SMS Received
- Waiting Days

The model uses `class_weight="balanced"` to address the imbalance between attended and no-show appointments.

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 58.62% |
| Precision | 30.38% |
| Recall | 80.65% |
| F1 Score | 44.13% |

## 🚦 Risk Classification

The model generates a no-show probability and classifies appointments into:

- 🟢 Low Risk: 0–30%
- 🟡 Medium Risk: 30–70%
- 🔴 High Risk: 70–100%

The model identified **10,361 high-risk appointments** using these thresholds.

## 📈 Key Findings

- Overall no-show rate: **20.26%**
- Total appointments: **106,987**
- Attended appointments: **85,307**
- No-show appointments: **21,680**
- No-show rates vary across different age groups.
- Waiting time is included as an important predictive feature.
- Risk-based prediction can help prioritize appointment-management activities.

## 📊 Power BI Dashboard

The project includes a Power BI-ready dataset and dashboard covering:

- Total appointments
- No-show rate
- SMS reminder analysis
- Age-group analysis
- Weekday analysis
- Risk distribution
- Waiting-time analysis
- Actual vs predicted outcomes
- Model performance

## 📁 Project Structure

```text
HEALTHCARE APPOINTMENT
│
├── healthcare_noshows.py
├── healthcare_noshows.csv
├── healthcare_predictions_for_powerbi.csv
├── overall_no_show.png
├── sms_no_show.png
├── age_no_show.png
├── weekday_no_show.png
├── powerbi_dashboard.pdf
└── README.md

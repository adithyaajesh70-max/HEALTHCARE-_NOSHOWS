import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load healthcare appointment dataset
df = pd.read_csv("data/healthcare_noshows.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData information:")
print(df.info())

print("\nShowed_up distribution:")
print(df["Showed_up"].value_counts())

print("\nShowed_up percentage:")
print(df["Showed_up"].value_counts(normalize=True) * 100)

# Create target variable
df["No_Show"] = (~df["Showed_up"]).astype(int)

print("\nNo_Show distribution:")
print(df["No_Show"].value_counts())

# Convert date columns to datetime
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

# Create appointment weekday
df["Appointment_Weekday"] = df["AppointmentDay"].dt.day_name()

# Create appointment month
df["Appointment_Month"] = df["AppointmentDay"].dt.month_name()

# Calculate waiting days
df["Waiting_Days"] = (
    df["AppointmentDay"] - df["ScheduledDay"]
).dt.days

print("\nNew date features:")
print(df[[
    "ScheduledDay",
    "AppointmentDay",
    "Appointment_Weekday",
    "Appointment_Month",
    "Waiting_Days"
]].head())

# No-show rate by weekday
weekday_analysis = (
    df.groupby("Appointment_Weekday")["No_Show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nNo-show rate by weekday:")
print(weekday_analysis)

# No-show rate by SMS reminder
sms_analysis = (
    df.groupby("SMS_received")["No_Show"]
    .mean()
    .mul(100)
)

print("\nNo-show rate by SMS reminder:")
print(sms_analysis)

# Create age groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[-1, 18, 30, 45, 60, 100],
    labels=[
        "0-18",
        "19-30",
        "31-45",
        "46-60",
        "61+"
    ]
)

# Calculate no-show rate by age group
age_analysis = (
    df.groupby("Age_Group", observed=True)["No_Show"]
    .mean()
    .mul(100)
)

print("\nNo-show rate by age group:")
print(age_analysis)

# ==============================
# VISUALIZATIONS
# ==============================

# 1. Overall No-Show Distribution
df["No_Show"].value_counts().plot(kind="bar")

plt.title("Appointment Attendance vs No-Show")
plt.xlabel("No-Show")
plt.ylabel("Number of Appointments")
plt.xticks([0, 1], ["Attended", "No-Show"], rotation=0)
plt.show()


# 2. No-Show Rate by SMS Reminder
sms_analysis.plot(kind="bar")

plt.title("No-Show Rate by SMS Reminder")
plt.xlabel("SMS Received")
plt.ylabel("No-Show Rate (%)")
plt.xticks([0, 1], ["No SMS", "SMS"], rotation=0)
plt.show()


# 3. No-Show Rate by Age Group
age_analysis.plot(kind="bar")

plt.title("No-Show Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("No-Show Rate (%)")
plt.xticks(rotation=0)
plt.show()


# 4. No-Show Rate by Weekday
weekday_analysis.plot(kind="bar")

plt.title("No-Show Rate by Appointment Weekday")
plt.xlabel("Weekday")
plt.ylabel("No-Show Rate (%)")
plt.xticks(rotation=45)
plt.show()

# ============================================================
# STEP 22: DECISION TREE MACHINE LEARNING MODEL
# ============================================================

# Select features for prediction
features = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "Waiting_Days"
]

# Input variables
X = df[features]

# Target variable
y = df["No_Show"]


# ============================================================
# STEP 23: SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# STEP 24: CREATE DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# ============================================================
# STEP 25: TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ============================================================
# STEP 26: MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# STEP 27: MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ============================================================
# STEP 28: CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# STEP 29: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# STEP 30: FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ============================================================
# STEP 31: FEATURE IMPORTANCE GRAPH
# ============================================================

importance.plot(
    x="Feature",
    y="Importance",
    kind="bar"
)

plt.title(
    "Feature Importance - Decision Tree"
)

plt.xlabel("Feature")

plt.ylabel("Importance")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.close()


# ============================================================
# STEP 32: PREDICT ALL PATIENTS
# ============================================================

df["Predicted_No_Show"] = model.predict(X)


# ============================================================
# STEP 33: NO-SHOW PROBABILITY
# ============================================================

df["No_Show_Probability"] = (
    model.predict_proba(X)[:, 1]
)


# ============================================================
# STEP 34: CREATE RISK LEVEL
# ============================================================

df["Risk_Level"] = pd.cut(
    df["No_Show_Probability"],
    bins=[0, 0.30, 0.70, 1],
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    include_lowest=True
)


# ============================================================
# STEP 35: DISPLAY RISK LEVEL
# ============================================================

print("\nRisk Level Distribution:")
print(
    df["Risk_Level"].value_counts()
)


# ============================================================
# STEP 36: SAVE FINAL PREDICTION DATASET
# ============================================================

df.to_csv(
    "predictions.csv",
    index=False
)

print("\nFinal prediction file created:")
print("predictions.csv")


print("\n" + "=" * 60)
print("MACHINE LEARNING PROCESS COMPLETED!")
print("=" * 60)




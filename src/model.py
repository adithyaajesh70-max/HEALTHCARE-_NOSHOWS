import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("healthcare_noshows.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# Create No-Show target
# True = attended
# False = did not attend

df["No_Show"] = (~df["Showed_up"]).astype(int)

print("\nNo-Show distribution:")
print(df["No_Show"].value_counts())


# Convert dates
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])


# Calculate waiting days
df["Waiting_Days"] = (
    df["AppointmentDay"] - df["ScheduledDay"]
).dt.days


# Select features
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

X = df[features]
y = df["No_Show"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

model = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")


# Prediction
y_pred = model.predict(X_test)


# Evaluation
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


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Feature importance
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


# Predict all appointments
df["Predicted_No_Show"] = model.predict(X)

df["No_Show_Probability"] = (
    model.predict_proba(X)[:, 1]
)


# Risk level
df["Risk_Level"] = pd.cut(
    df["No_Show_Probability"],
    bins=[0, 0.30, 0.70, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("\nRisk Level Distribution:")
print(df["Risk_Level"].value_counts())


# Save predictions
df.to_csv(
    "predictions.csv",
    index=False
)

print("\n================================")
print("predictions.csv CREATED!")
print("================================")
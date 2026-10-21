
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset and trained model
df = pd.read_csv("dataset/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

model = joblib.load("model/loan_model.pkl")
education_encoder = joblib.load("model/education_encoder.pkl")
employment_encoder = joblib.load("model/employment_encoder.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

# 2. Clean text values
for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].astype(str).str.strip()

# 3. Remove ID
df = df.drop(columns=["loan_id"])

# 4. Encode values using the SAME encoders used for training
df["education"] = education_encoder.transform(df["education"])
df["self_employed"] = employment_encoder.transform(df["self_employed"])
df["loan_status"] = target_encoder.transform(df["loan_status"])

# 5. Separate input features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# 6. Use the same test split as train_model.py
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Match the trained model's expected feature order
if hasattr(model, "feature_names_in_"):
    X_test = X_test[list(model.feature_names_in_)]

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
print("\n========== MODEL EVALUATION ==========")
print(f"Test samples: {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(target_encoder.classes_)),
    target_names=target_encoder.classes_,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(target_encoder.classes_))
))

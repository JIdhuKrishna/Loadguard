import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/loan_approval_dataset.csv")

print("Dataset loaded successfully!")


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

print("\nColumns after cleaning:")
print(df.columns)


# ==========================================
# 3. CLEAN TEXT VALUES
# ==========================================

df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()


# ==========================================
# 4. REMOVE LOAN ID
# ==========================================

df = df.drop("loan_id", axis=1)


# ==========================================
# 5. ENCODE CATEGORICAL DATA
# ==========================================

education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["education"] = education_encoder.fit_transform(df["education"])

df["self_employed"] = employment_encoder.fit_transform(
    df["self_employed"]
)

df["loan_status"] = target_encoder.fit_transform(
    df["loan_status"]
)


# ==========================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("loan_status", axis=1)

y = df["loan_status"]


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ==========================================
# 10. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 11. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_
))


# ==========================================
# 12. CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 13. SAVE MODEL AND ENCODERS
# ==========================================

joblib.dump(model, "model/loan_model.pkl")

joblib.dump(
    education_encoder,
    "model/education_encoder.pkl"
)

joblib.dump(
    employment_encoder,
    "model/employment_encoder.pkl"
)

joblib.dump(
    target_encoder,
    "model/target_encoder.pkl"
)


print("\nAll model files saved successfully!")
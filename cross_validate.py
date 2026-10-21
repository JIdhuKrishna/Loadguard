
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "loan_approval_dataset.csv"

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].astype("string").str.strip()

df = df.drop(columns=["loan_id"], errors="ignore")
df = df.dropna().copy()

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Identify feature types
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\n5-Fold Cross-Validation Results")

for metric in scoring:
    scores = results[f"test_{metric}"]

    print(f"\n{metric.title()}:")
    print("Individual folds:", scores.round(4))
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std deviation: {scores.std():.4f}")

train_accuracy = results["train_accuracy"]
test_accuracy = results["test_accuracy"]

print("\nTraining accuracy per fold:", train_accuracy.round(4))
print("Validation accuracy per fold:", test_accuracy.round(4))
print(f"Mean training accuracy: {train_accuracy.mean():.4f}")
print(f"Mean validation accuracy: {test_accuracy.mean():.4f}")

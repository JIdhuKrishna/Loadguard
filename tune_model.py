
from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "dataset" / "loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].astype("string").str.strip()

df = df.drop(columns=["loan_id"], errors="ignore").dropna()

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Tune only on the training set
parameters = {
    "model__n_estimators": [100],
    "model__max_depth": [None, 10, 15, 20],
    "model__min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipeline,
    parameters,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.4f}")
print(f"Mean training accuracy for best setting: "
      f"{search.cv_results_['mean_train_score'][search.best_index_]:.4f}")

best_model = search.best_estimator_
predictions = best_model.predict(X_test)

print(f"\nHeld-out test accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(y_test, predictions))

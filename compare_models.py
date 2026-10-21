
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load and clean the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "loan_approval_dataset.csv"

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# Remove extra spaces from text columns
for column in ["education", "self_employed", "loan_status"]:
    df[column] = df[column].astype(str).str.strip()

# Standardize target labels
df["loan_status"] = df["loan_status"].str.title()

# Remove identifier; it should not be used for prediction
df = df.drop(columns=["loan_id"])

# Remove rows with missing values for this baseline comparison
df = df.dropna().copy()

# Separate features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Display class balance
print("\n--- Actual dataset class balance ---")
print(y.value_counts())
print("\nClass percentages:")
print((y.value_counts(normalize=True) * 100).round(2))

# 2. Split the data identically for all models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess features consistently
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 3. Define the three algorithms
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=None
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

results = []
confusion_matrices = {}

# 4. Train and evaluate each model
for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Metrics use Approved as the positive class
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        pos_label="Approved", zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        pos_label="Approved", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        pos_label="Approved", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision (Approved)": precision,
        "Recall (Approved)": recall,
        "F1-score (Approved)": f1
    })

    # Keep a consistent label order
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Rejected", "Approved"]
    )
    confusion_matrices[name] = cm

    print(f"\n{'=' * 50}")
    print(f"MODEL: {name}")
    print(f"{'=' * 50}")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["Approved", "Rejected"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(pd.DataFrame(
        cm,
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"]
    ))

    # Diagnose over-prediction of Approved
    print("\nPrediction distribution:")
    print(pd.Series(predictions).value_counts())

    approved_rate = (predictions == "Approved").mean() * 100
    actual_approved_rate = (y_test == "Approved").mean() * 100

    print(f"Predicted Approved: {approved_rate:.2f}%")
    print(f"Actually Approved in test set: {actual_approved_rate:.2f}%")

# 5. Compare all model metrics
results_df = pd.DataFrame(results).set_index("Model")

print("\n\n--- MODEL COMPARISON ---")
print(results_df.round(4))

# Save results for your report
results_df.to_csv(BASE_DIR / "model_comparison_results.csv")

# 6. Plot metric comparison
results_df.plot(kind="bar", figsize=(12, 6))
plt.title("Loan Approval Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(BASE_DIR / "model_comparison.png", dpi=150)
plt.show()

# 7. Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrices.png", dpi=150)
plt.show()

print("\nSaved:")
print("- model_comparison_results.csv")
print("- model_comparison.png")
print("- confusion_matrices.png")

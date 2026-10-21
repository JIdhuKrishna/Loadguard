
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "loan_approval_dataset.csv"

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].astype("string").str.strip()

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate complete rows:", df.duplicated().sum())

if "loan_id" in df.columns:
    print("Duplicate loan IDs:", df["loan_id"].duplicated().sum())

print("\nTarget distribution:")
print(df["loan_status"].value_counts())

# Check whether any feature column is an exact copy
# of the target after converting both to strings.
target = df["loan_status"].astype("string")

print("\nFeatures identical to target:")
found = False

for col in df.columns:
    if col == "loan_status":
        continue

    feature = df[col].astype("string")
    if feature.equals(target):
        print("Possible leakage:", col)
        found = True

if not found:
    print("No feature is an exact copy of the target.")

# Check whether the target is constant within identical
# combinations of all input features.
feature_cols = [
    c for c in df.columns
    if c not in ["loan_status", "loan_id"]
]

if feature_cols:
    grouped = df.groupby(feature_cols, dropna=False)["loan_status"].nunique()
    conflicts = int((grouped > 1).sum())
    print(
        "\nIdentical feature combinations with conflicting labels:",
        conflicts
    )

print(
    "\nNote: these checks can flag suspicious data, "
    "but cannot prove that leakage is absent."
)

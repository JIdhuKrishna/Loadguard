import pandas as pd

df = pd.read_csv("dataset/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLoan Status:")
print(df["loan_status"].value_counts())
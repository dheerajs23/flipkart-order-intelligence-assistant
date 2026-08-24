import pandas as pd

# Load dataset
df = pd.read_csv("orders_dataset.csv")

print("========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== OVERALL RETURN RATE ==========")
print(df["returned"].mean())

print("\n========== RETURN RATE BY CATEGORY ==========")
print(
    df.groupby("product_category")["returned"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== RETURN RATE BY PAYMENT METHOD ==========")
print(
    df.groupby("payment_method")["returned"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== CATEGORY COUNTS ==========")
print(df["product_category"].value_counts())

print("\n========== PAYMENT METHOD COUNTS ==========")
print(df["payment_method"].value_counts())

print("\n========== DATA TYPES ==========")
print(df.dtypes)
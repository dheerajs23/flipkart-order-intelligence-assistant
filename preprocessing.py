import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("orders_dataset.csv")

# 2. Handle missing ratings
df["rating_given"] = df["rating_given"].fillna(df["rating_given"].median())

# 3. Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=["product_category", "payment_method"],
    drop_first=True
)

# 4. Separate features (X) and target (y)
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Display results
print("========== PREPROCESSING RESULT ==========")
print("Original rows:", len(df))
print("Number of features:", X.shape[1])

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\ny_train distribution:")
print(y_train.value_counts())

print("\ny_test distribution:")
print(y_test.value_counts())

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())

print("\nFeature columns:")
print(X.columns.tolist())
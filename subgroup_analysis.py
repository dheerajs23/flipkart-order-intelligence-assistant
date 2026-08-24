import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("orders_dataset.csv")


# ==========================================
# 2. Features and target
# ==========================================

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]


# ==========================================
# 3. Feature types
# ==========================================

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]

categorical_features = [
    "product_category",
    "payment_method"
]


# ==========================================
# 4. Preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 5. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Logistic Regression
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000
            )
        )
    ]
)

model.fit(X_train, y_train)


# ==========================================
# 7. Predictions using tuned threshold
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.44

predictions = (
    probabilities >= threshold
).astype(int)


# ==========================================
# 8. Create test dataframe
# ==========================================

results = X_test.copy()

results["actual_returned"] = y_test
results["predicted_returned"] = predictions


# ==========================================
# 9. Category subgroup analysis
# ==========================================

print("\n========== PRODUCT CATEGORY PERFORMANCE ==========")

category_results = []

for category in sorted(results["product_category"].unique()):

    group = results[
        results["product_category"] == category
    ]

    precision = precision_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    recall = recall_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    f1 = f1_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    category_results.append({
        "product_category": category,
        "orders": len(group),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


category_df = pd.DataFrame(category_results)

print(
    category_df.to_string(index=False)
)


# ==========================================
# 10. Payment subgroup analysis
# ==========================================

print("\n========== PAYMENT METHOD PERFORMANCE ==========")

payment_results = []

for payment in sorted(results["payment_method"].unique()):

    group = results[
        results["payment_method"] == payment
    ]

    precision = precision_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    recall = recall_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    f1 = f1_score(
        group["actual_returned"],
        group["predicted_returned"],
        zero_division=0
    )

    payment_results.append({
        "payment_method": payment,
        "orders": len(group),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


payment_df = pd.DataFrame(payment_results)

print(
    payment_df.to_string(index=False)
)


# ==========================================
# 11. Save results
# ==========================================

category_df.to_csv(
    "category_subgroup_performance.csv",
    index=False
)

payment_df.to_csv(
    "payment_subgroup_performance.csv",
    index=False
)

print("\nSubgroup analysis completed.")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


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
# 4. Numeric preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 5. Categorical preprocessing
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 6. Combined preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 7. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. Logistic Regression pipeline
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


# ==========================================
# 9. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 10. Default threshold = 0.5
# ==========================================

y_probability = model.predict_proba(X_test)[:, 1]

y_pred_default = (y_probability >= 0.5).astype(int)


accuracy = accuracy_score(y_test, y_pred_default)
precision = precision_score(y_test, y_pred_default)
recall = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)
roc_auc = roc_auc_score(y_test, y_probability)


print("========== LOGISTIC REGRESSION ==========")

print("\nDefault threshold: 0.50")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# 11. Threshold sweep
# ==========================================

thresholds = np.arange(0.10, 0.901, 0.01)

results = []

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    precision_t = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall_t = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1_t = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision_t,
        "recall": recall_t,
        "f1": f1_t
    })


threshold_results = pd.DataFrame(results)


# ==========================================
# 12. Find best threshold
# ==========================================

best_row = threshold_results.loc[
    threshold_results["f1"].idxmax()
]

best_threshold = best_row["threshold"]
best_precision = best_row["precision"]
best_recall = best_row["recall"]
best_f1 = best_row["f1"]


print("\n========== THRESHOLD TUNING ==========")

print(
    "Best threshold:",
    round(best_threshold, 2)
)

print(
    "Precision:",
    round(best_precision, 4)
)

print(
    "Recall   :",
    round(best_recall, 4)
)

print(
    "F1-score :",
    round(best_f1, 4)
)


# ==========================================
# 13. Compare default and tuned threshold
# ==========================================

print("\n========== COMPARISON ==========")

print(
    "Default recall:",
    round(recall, 4)
)

print(
    "Tuned recall  :",
    round(best_recall, 4)
)

print(
    "Recall improvement:",
    round(best_recall - recall, 4)
)

print(
    "Default precision:",
    round(precision, 4)
)

print(
    "Tuned precision  :",
    round(best_precision, 4)
)

print(
    "Precision change:",
    round(best_precision - precision, 4)
)


# ==========================================
# 14. Save threshold results
# ==========================================

threshold_results.to_csv(
    "logistic_threshold_results.csv",
    index=False
)

print("\nThreshold results saved to:")
print("logistic_threshold_results.csv")
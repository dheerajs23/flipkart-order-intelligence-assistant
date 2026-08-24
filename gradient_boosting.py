import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
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
# 6. ColumnTransformer
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
# 8. Gradient Boosting model
# ==========================================

classifier = GradientBoostingClassifier(
    learning_rate=0.03,
    n_estimators=300,
    max_depth=2,
    random_state=42
)


# ==========================================
# 9. Complete pipeline
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ==========================================
# 10. Train
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 11. Predictions
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 12. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 13. Results
# ==========================================

print("========== GRADIENT BOOSTING ==========")

print("\nParameters:")
print("learning_rate:", 0.03)
print("n_estimators :", 300)
print("max_depth    :", 2)

print("\nEvaluation:")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("\n========== PROBABILITY ANALYSIS ==========")

print("Minimum probability:", round(y_probability.min(), 4))
print("Maximum probability:", round(y_probability.max(), 4))
print("Mean probability:", round(y_probability.mean(), 4))

print("\nProbability percentiles:")
print(
    pd.Series(y_probability).describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nNumber of predictions >= 0.50:")
print((y_probability >= 0.50).sum())
# ==========================================
# 15. Threshold sweep
# ==========================================

import numpy as np

thresholds = np.arange(0.10, 0.501, 0.01)

threshold_results = []

for threshold in thresholds:

    threshold_pred = (
        y_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": threshold_precision,
        "recall": threshold_recall,
        "f1": threshold_f1
    })


threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print("\n========== GRADIENT BOOSTING THRESHOLD TUNING ==========")

print("Best threshold:", round(best_row["threshold"], 2))
print("Precision:", round(best_row["precision"], 4))
print("Recall:", round(best_row["recall"], 4))
print("F1-score:", round(best_row["f1"], 4))

threshold_df.to_csv(
    "gradient_boosting_threshold_results.csv",
    index=False
)
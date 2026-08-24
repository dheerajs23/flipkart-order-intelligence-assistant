import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


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
# 7. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. Random Forest
# ==========================================

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)


# ==========================================
# 9. Full Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ]
)


# ==========================================
# 10. Parameter Grid
# ==========================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}


# ==========================================
# 11. Stratified 5-Fold CV
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 12. GridSearchCV
# ==========================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=False
)


# ==========================================
# 13. Train GridSearch
# ==========================================

print("Training Random Forest GridSearchCV...")

grid_search.fit(X_train, y_train)


# ==========================================
# 14. Results
# ==========================================

print("\n========== RANDOM FOREST ==========")

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validated ROC-AUC:")
print(round(grid_search.best_score_, 4))


# ==========================================
# 15. Test-set ROC-AUC
# ==========================================

best_model = grid_search.best_estimator_

y_probability = best_model.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nTest-set ROC-AUC:")
print(round(test_roc_auc, 4))


# ==========================================
# 16. Compare CV and test ROC-AUC
# ==========================================

difference = abs(
    grid_search.best_score_ - test_roc_auc
)

print("\nDifference between CV and test ROC-AUC:")
print(round(difference, 4))


# ==========================================
# 17. Acceptance criteria
# ==========================================

print("\n========== ACCEPTANCE CHECK ==========")

if grid_search.best_score_ >= 0.58:
    print("CV ROC-AUC >= 0.58: PASS")
else:
    print("CV ROC-AUC >= 0.58: FAIL")

if difference <= 0.05:
    print("CV/Test ROC-AUC difference <= 0.05: PASS")
else:
    print("CV/Test ROC-AUC difference <= 0.05: FAIL")
# ==========================================
# 18. Feature Importance
# ==========================================

importances = best_model.named_steps["classifier"].feature_importances_

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n========== TOP 5 FEATURE IMPORTANCE ==========")

print(
    feature_importance_df.head(5).to_string(index=False)
)


# ==========================================
# 19. Save feature importance
# ==========================================

feature_importance_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("random_forest_feature_importance.csv")
# ==========================================
# 20. Permutation Importance
# ==========================================

from sklearn.inspection import permutation_importance


print("\n========== PERMUTATION IMPORTANCE ==========")

permutation_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
})

permutation_df = permutation_df.sort_values(
    by="importance_mean",
    ascending=False
)

print(
    permutation_df.head(10).to_string(index=False)
)

permutation_df.to_csv(
    "random_forest_permutation_importance.csv",
    index=False
)

print("\nPermutation importance saved to:")
print("random_forest_permutation_importance.csv")
# ==========================================
# 21. Random Forest classification metrics
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n========== RANDOM FOREST CLASSIFICATION ==========")

# Default threshold
rf_pred_default = (
    y_probability >= 0.50
).astype(int)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred_default
)

rf_precision = precision_score(
    y_test,
    rf_pred_default,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred_default,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred_default,
    zero_division=0
)

print("\nDefault threshold: 0.50")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1-score :", round(rf_f1, 4))


# ==========================================
# 22. Random Forest threshold tuning
# ==========================================

thresholds = [i / 100 for i in range(10, 91)]

rf_threshold_results = []

for threshold in thresholds:

    rf_pred = (
        y_probability >= threshold
    ).astype(int)

    rf_threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_test,
            rf_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            rf_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            rf_pred,
            zero_division=0
        )
    })


rf_threshold_df = pd.DataFrame(
    rf_threshold_results
)

best_rf_row = rf_threshold_df.loc[
    rf_threshold_df["f1"].idxmax()
]

print("\n========== RANDOM FOREST THRESHOLD TUNING ==========")

print(
    "Best threshold:",
    round(best_rf_row["threshold"], 2)
)

print(
    "Precision:",
    round(best_rf_row["precision"], 4)
)

print(
    "Recall   :",
    round(best_rf_row["recall"], 4)
)

print(
    "F1-score :",
    round(best_rf_row["f1"], 4)
)

rf_threshold_df.to_csv(
    "random_forest_threshold_results.csv",
    index=False
)

print("\nThreshold results saved to:")
print("random_forest_threshold_results.csv")
# ==========================================
# SAVE FINAL RANDOM FOREST MODEL
# ==========================================

final_rf_model = grid_search.best_estimator_

joblib.dump(
    final_rf_model,
    "models/return_risk_model.pkl"
)

print("\nFinal Random Forest model saved to:")
print("models/return_risk_model.pkl")

print("Random Forest F1-maximizing threshold (t*_rf):")
print(round(best_rf_row["threshold"], 2))
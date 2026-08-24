import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib


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
# 5. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Final Logistic Regression model
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
# 7. Train final model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. Save model
# ==========================================

joblib.dump(
    model,
    "return_risk_model.joblib"
)

print("Final model trained and saved.")

print("Model file:")
print("return_risk_model.joblib")


# ==========================================
# 9. Prediction function
# ==========================================

def predict_return_risk(order):

    order_df = pd.DataFrame([order])

    probability = model.predict_proba(
        order_df
    )[0, 1]

    threshold = 0.44

    prediction = int(
        probability >= threshold
    )

    if probability >= 0.60:
        risk_level = "HIGH"

    elif probability >= 0.44:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "return_probability": round(
            float(probability),
            4
        ),
        "threshold": threshold,
        "predicted_return": prediction,
        "risk_level": risk_level
    }


# ==========================================
# 10. Test prediction
# ==========================================

sample_order = X_test.iloc[0].to_dict()

result = predict_return_risk(
    sample_order
)

print("\n========== SAMPLE PREDICTION ==========")

print("Order:")
print(sample_order)

print("\nPrediction:")
print(result)
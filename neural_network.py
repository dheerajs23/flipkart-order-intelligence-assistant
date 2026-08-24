import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. Reproducibility
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv("orders_dataset.csv")


# ==========================================
# 3. Features and target
# ==========================================

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]


# ==========================================
# 4. Feature types
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
# 5. Numeric preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 6. Categorical preprocessing
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 7. ColumnTransformer
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 8. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. Fit preprocessing only on training data
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 10. Neural Network
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# 11. Adam optimizer
# ==========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)


model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 12. Early stopping
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# 13. Train
# ==========================================

print("========== NEURAL NETWORK TRAINING ==========")

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)


# ==========================================
# 14. Test probabilities
# ==========================================

y_probability = model.predict(
    X_test_processed,
    verbose=0
).ravel()


# ==========================================
# 15. Default threshold = 0.5
# ==========================================

y_pred = (
    y_probability >= 0.50
).astype(int)


# ==========================================
# 16. Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
# 17. Results
# ==========================================

print("\n========== NEURAL NETWORK RESULTS ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nEpochs actually trained:")
print(len(history.history["loss"]))
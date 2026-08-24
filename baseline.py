import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("orders_dataset.csv")


# ==========================================
# 2. Separate X and y
# ==========================================

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]


# ==========================================
# 3. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Dummy Classifier
# ==========================================

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

y_pred = dummy_model.predict(X_test)


# ==========================================
# 5. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


# ==========================================
# 6. Results
# ==========================================

print("========== DUMMY CLASSIFIER ==========")

print("Accuracy:", round(accuracy, 4))

print("F1-score (returned=1):", round(f1, 4))

print("\nPredicted class distribution:")
print(pd.Series(y_pred).value_counts())
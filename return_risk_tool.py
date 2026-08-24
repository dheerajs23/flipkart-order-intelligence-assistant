import joblib
import pandas as pd


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/return_risk_model.pkl"


EXPECTED_FEATURES = [
    "product_category",
    "price_inr",
    "discount_pct",
    "payment_method",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]


# ==========================================
# Load trained model
# ==========================================

print("Loading return-risk model...")

model = joblib.load(
    MODEL_PATH
)

print("Return-risk model loaded successfully.")


# ==========================================
# Return-risk prediction tool
# ==========================================

def check_return_risk(order_data):

    if not isinstance(order_data, dict):
        raise TypeError(
            "order_data must be a dictionary."
        )

    # Check required fields
    missing_fields = [
        field
        for field in EXPECTED_FEATURES
        if field not in order_data
    ]

    if missing_fields:

        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )

    # Keep only expected features
    input_data = {
        field: order_data[field]
        for field in EXPECTED_FEATURES
    }

    input_df = pd.DataFrame(
        [input_data]
    )

    # Predict probability
    probability = model.predict_proba(
        input_df
    )[0][1]

    probability = float(
        probability
    )

    # Risk level
    if probability >= 0.50:
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    return {
        "return_probability": round(
            probability,
            4
        ),

        "return_probability_percent": round(
            probability * 100,
            2
        ),

        "risk_level": risk_level
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    test_order = {

        "product_category": "Fashion",

        "price_inr": 1500,

        "discount_pct": 20,

        "payment_method": "COD",

        "customer_tenure_days": 300,

        "num_previous_orders": 10,

        "num_previous_returns": 2,

        "delivery_distance_km": 8,

        "delivery_days": 4,

        "is_weekend_order": 0,

        "rating_given": 4
    }

    print(
        "\n========== RETURN RISK TOOL TEST =========="
    )

    result = check_return_risk(
        test_order
    )

    print(
        "Return probability:",
        result["return_probability_percent"],
        "%"
    )

    print(
        "Risk level:",
        result["risk_level"]
    )
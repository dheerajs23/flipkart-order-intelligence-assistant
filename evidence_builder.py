import pandas as pd


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("orders_dataset.csv")


# ==========================================
# Dataset-level statistics
# ==========================================

overall_return_rate = df["returned"].mean()

category_return_rates = (
    df.groupby("product_category")["returned"]
    .mean()
    .to_dict()
)

payment_return_rates = (
    df.groupby("payment_method")["returned"]
    .mean()
    .to_dict()
)


# ==========================================
# Evidence builder
# ==========================================

def build_evidence(order, prediction):

    evidence = {
        "prediction": prediction,
        "order_facts": {},
        "dataset_evidence": [],
        "cautions": []
    }

    # --------------------------------------
    # Actual order facts
    # --------------------------------------

    evidence["order_facts"] = {
        "product_category": order["product_category"],
        "price_inr": order["price_inr"],
        "discount_pct": order["discount_pct"],
        "payment_method": order["payment_method"],
        "customer_tenure_days": order["customer_tenure_days"],
        "num_previous_orders": order["num_previous_orders"],
        "num_previous_returns": order["num_previous_returns"],
        "delivery_distance_km": order["delivery_distance_km"],
        "delivery_days": order["delivery_days"],
        "is_weekend_order": order["is_weekend_order"],
        "rating_given": order["rating_given"]
    }

    # --------------------------------------
    # Overall return rate
    # --------------------------------------

    evidence["dataset_evidence"].append({
        "type": "overall_return_rate",
        "value": round(
            overall_return_rate,
            4
        )
    })

    # --------------------------------------
    # Category evidence
    # --------------------------------------

    category = order["product_category"]

    if category in category_return_rates:

        evidence["dataset_evidence"].append({
            "type": "category_return_rate",
            "category": category,
            "value": round(
                category_return_rates[category],
                4
            )
        })

    # --------------------------------------
    # Payment evidence
    # --------------------------------------

    payment = order["payment_method"]

    if payment in payment_return_rates:

        evidence["dataset_evidence"].append({
            "type": "payment_return_rate",
            "payment_method": payment,
            "value": round(
                payment_return_rates[payment],
                4
            )
        })

    # --------------------------------------
    # Important cautions
    # --------------------------------------

    evidence["cautions"] = [
        "Observed associations do not prove causation.",
        "The return probability is a model estimate, not a certainty.",
        "The LLM must not invent reasons that are not present in the evidence."
    ]

    return evidence


# ==========================================
# Test
# ==========================================

sample_order = df.drop(
    columns=["returned", "order_id"]
).iloc[0].to_dict()

sample_prediction = {
    "return_probability": 0.5651,
    "threshold": 0.44,
    "predicted_return": 1,
    "risk_level": "MEDIUM"
}

evidence = build_evidence(
    sample_order,
    sample_prediction
)

print("\n========== EVIDENCE ==========")

print(evidence)
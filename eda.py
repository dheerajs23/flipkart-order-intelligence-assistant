import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("orders_dataset.csv")


# ==========================================
# 1. Return Rate by Product Category
# ==========================================

category_return_rate = (
    df.groupby("product_category")["returned"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
category_return_rate.plot(kind="bar")

plt.title("Return Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Return Rate")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("return_rate_by_category.png")
plt.show()


# ==========================================
# 2. Return Rate by Payment Method
# ==========================================

payment_return_rate = (
    df.groupby("payment_method")["returned"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
payment_return_rate.plot(kind="bar")

plt.title("Return Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Return Rate")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("return_rate_by_payment.png")
plt.show()


# ==========================================
# 3. Price Distribution by Return Status
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="price_inr",
    by="returned"
)

plt.title("Price Distribution by Return Status")
plt.suptitle("")
plt.xlabel("Returned (0 = No, 1 = Yes)")
plt.ylabel("Price (INR)")
plt.tight_layout()

plt.savefig("price_by_return_status.png")
plt.show()


# ==========================================
# 4. Missing Rating Pattern
# ==========================================

df["rating_missing"] = df["rating_given"].isna()

missing_by_payment = (
    df.groupby("payment_method")["rating_missing"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
missing_by_payment.plot(kind="bar")

plt.title("Missing Rating Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Missing Rating Rate")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("missing_rating_by_payment.png")
plt.show()


print("\nEDA completed successfully.")

print("\nMissing rating rate by payment method:")
print(missing_by_payment)

print("\nReturn rate by category:")
print(category_return_rate)

print("\nReturn rate by payment method:")
print(payment_return_rate)
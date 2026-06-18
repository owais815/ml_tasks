# ==============================================================
# DAY 6 — FEATURE ENGINEERING PART 1
# Course: ML Fundamentals | Week 2
# ==============================================================

# FEATURE ENGINEERING = creating new, more useful columns from existing ones.
# A "feature" = one input column that the ML model uses to make predictions.
# Better features → smarter models. This is often the most valuable skill in ML!

import pandas as pd
import numpy as np

print("=" * 60)
print("DAY 6: FEATURE ENGINEERING PART 1")
print("=" * 60)

df = pd.read_csv("../data/day05_no_outliers.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("Columns:", df.columns.tolist())


# ==============================================================
# SECTION 1: WHY FEATURE ENGINEERING MATTERS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: WHY FEATURE ENGINEERING?")
print("=" * 60)

# Raw data rarely has the PERFECT format for ML models.
# Example: a date "2024-01-15" is just text — useless for math.
# But "month=January", "day_of_week=Monday", "is_weekend=False" — USEFUL!

# Feature engineering can:
#   1. Create more informative inputs
#   2. Capture relationships the raw data hides
#   3. Dramatically improve model accuracy


# ==============================================================
# SECTION 2: DATE / TIME FEATURE EXTRACTION
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: DATE & TIME FEATURES")
print("=" * 60)

# Convert the purchase_date string into a proper datetime object.
# pd.to_datetime() understands date strings and parses them.
df["purchase_date"] = pd.to_datetime(df["purchase_date"])

# Now we can extract all sorts of useful things from the date!

# Year the purchase happened
df["purchase_year"]  = df["purchase_date"].dt.year

# Month (1=January, 12=December) — helps spot seasonal trends
df["purchase_month"] = df["purchase_date"].dt.month

# Day of week (0=Monday, 6=Sunday) — are people buying on weekends?
df["purchase_day_of_week"] = df["purchase_date"].dt.dayofweek

# Is the purchase on a weekend? (Saturday=5, Sunday=6)
df["is_weekend"] = df["purchase_day_of_week"].isin([5, 6]).astype(int)
# .astype(int) converts True→1 and False→0

# Quarter of the year (1–4) — useful for business/seasonal analysis
df["purchase_quarter"] = df["purchase_date"].dt.quarter

print("\nDate features extracted (first 5 rows):")
date_cols = ["purchase_date","purchase_year","purchase_month",
             "purchase_day_of_week","is_weekend","purchase_quarter"]
print(df[date_cols].head())


# ==============================================================
# SECTION 3: MATHEMATICAL / RATIO FEATURES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: RATIO & INTERACTION FEATURES")
print("=" * 60)

# RATIO: Combine two numbers to create a meaningful relationship

# Discount amount in £ (how much was saved)
df["discount_amount_gbp"] = df["purchase_amount"] * df["discount_pct"] / 100

# Final price after discount
df["final_price"] = df["purchase_amount"] - df["discount_amount_gbp"]

# Income-to-purchase ratio: how much of their income went to this purchase?
# Add a small epsilon (0.001) to avoid division by zero
df["spend_to_income_ratio"] = df["final_price"] / (df["income"] + 0.001)
df["spend_to_income_ratio"] = df["spend_to_income_ratio"].round(4)

# Purchase rate: purchases per year (loyalty indicator)
df["purchases_per_year"] = df["num_purchases_last_year"]   # already a rate

# Average spend per historical purchase
df["avg_historical_spend"] = df["final_price"] / (df["num_purchases_last_year"] + 1)
# +1 to avoid division by zero for customers with 0 prior purchases

print("\nNew ratio features (first 5 rows):")
ratio_cols = ["customer_id","purchase_amount","discount_pct",
              "discount_amount_gbp","final_price","spend_to_income_ratio"]
print(df[ratio_cols].head())

print("\nAverage historical spend (first 5 rows):")
print(df[["customer_id","num_purchases_last_year","avg_historical_spend"]].head())


# ==============================================================
# SECTION 4: DOMAIN-DRIVEN FEATURES (BUSINESS LOGIC)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: DOMAIN-DRIVEN FEATURES")
print("=" * 60)

# These features require BUSINESS KNOWLEDGE to create.
# They encode human expertise into the dataset.

# Is this a HIGH-VALUE purchase? (above the 75th percentile)
high_value_threshold = df["purchase_amount"].quantile(0.75)
df["is_high_value_purchase"] = (df["purchase_amount"] > high_value_threshold).astype(int)
print(f"\nHigh-value threshold (75th percentile): £{high_value_threshold:.2f}")
print(f"High-value purchases: {df['is_high_value_purchase'].sum()}")

# Is this customer a "LOYAL" customer? (more than 10 purchases last year)
df["is_loyal_customer"] = (df["num_purchases_last_year"] > 10).astype(int)
print(f"\nLoyal customers (>10 purchases/year): {df['is_loyal_customer'].sum()}")

# INCOME BRACKET (categorical grouping of income)
# pd.cut() divides a continuous variable into bins
income_bins   = [0, 20000, 35000, 55000, 75000, float("inf")]
income_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
df["income_bracket"] = pd.cut(df["income"], bins=income_bins, labels=income_labels)
print("\nIncome brackets distribution:")
print(df["income_bracket"].value_counts())


# ==============================================================
# SECTION 5: TEXT-BASED FEATURES (SIMPLE EXAMPLE)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: BASIC TEXT FEATURES")
print("=" * 60)

# Although our dataset doesn't have free text, let's demonstrate
# simple text features using the 'city' column.

# String length — could be useful if you had review text, etc.
df["city_name_length"] = df["city"].str.len()

# Is the city London? (binary flag for a specific important category)
df["is_london"] = (df["city"] == "London").astype(int)

print("\nText-derived features (first 8 rows):")
print(df[["customer_id","city","city_name_length","is_london"]].head(8))


# ==============================================================
# SECTION 6: CORRELATION ANALYSIS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: VALIDATE NEW FEATURES WITH CORRELATION")
print("=" * 60)

# CORRELATION tells us how closely related two numeric columns are.
# Values range from -1 to +1:
#   +1 = perfectly positively correlated (one goes up → other goes up)
#   -1 = perfectly negatively correlated (one goes up → other goes down)
#    0 = no relationship

# Which features are most related to purchase_amount?
numeric_cols = ["purchase_amount","final_price","income",
                "age","discount_pct","num_purchases_last_year",
                "spend_to_income_ratio","avg_historical_spend",
                "is_high_value_purchase","is_loyal_customer","is_member"]

corr_with_target = df[numeric_cols].corr()["purchase_amount"].drop("purchase_amount")
print("\nCorrelation of features with 'purchase_amount' (sorted):")
print(corr_with_target.sort_values(ascending=False).round(4))

# Features with high absolute correlation are the most informative for ML!


# ==============================================================
# SECTION 7: SAVE ENGINEERED DATASET
# ==============================================================

output_path = "../data/day06_engineered.csv"
df.to_csv(output_path, index=False)
print(f"\nEngineered dataset saved → {output_path}")
print(f"New shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"New features added: {df.shape[1] - 16}")   # Original had ~16 cols


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 6 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Feature engineering creates new informative columns from existing ones")
print("✓ Date features: year, month, day_of_week, is_weekend, quarter")
print("✓ Ratio features: combine two columns into a meaningful ratio")
print("✓ Domain features: encode business/domain knowledge as binary flags")
print("✓ pd.cut() creates categorical bins from continuous numbers")
print("✓ Validate new features using correlation with the target")
print("\nTomorrow: Feature Engineering Part 2 — Encoding Categorical Variables!")

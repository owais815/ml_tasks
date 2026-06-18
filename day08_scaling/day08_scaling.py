# ==============================================================
# DAY 8 — FEATURE SCALING
# Course: ML Fundamentals | Week 3
# ==============================================================

# SCALING = adjusting the RANGE of numbers so all features are comparable.
#
# Problem: 'age' is 20–60, but 'income' is 15,000–95,000.
# Without scaling, the model thinks income is 1,000× more important than age
# just because the numbers are bigger — even if age matters just as much!
#
# Scaling fixes this so every feature has an equal say.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

print("=" * 60)
print("DAY 8: FEATURE SCALING")
print("=" * 60)

df = pd.read_csv("../data/day07_encoded.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# We'll demonstrate scaling on a small subset of numeric columns
numeric_demo_cols = ["age", "income", "purchase_amount", "customer_rating", "num_purchases_last_year"]
df_demo = df[numeric_demo_cols].copy()

print("\nBefore scaling — ranges are very different:")
print(df_demo.describe().round(2))

import os
os.makedirs("charts", exist_ok=True)


# ==============================================================
# SECTION 1: WHY SCALING MATTERS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: WHY DOES SCALING MATTER?")
print("=" * 60)

# Algorithms AFFECTED by scaling (distance or gradient-based):
#   - KNN (K-Nearest Neighbours) — measures distance
#   - Linear/Logistic Regression — gradient descent
#   - SVM (Support Vector Machine)
#   - Neural Networks
#   - PCA (Principal Component Analysis)

# Algorithms NOT affected by scaling:
#   - Decision Trees
#   - Random Forests
#   - XGBoost / LightGBM
#   (Tree-based models split on thresholds — scale doesn't matter)

print("MUST scale:   KNN, Linear Regression, SVM, Neural Networks, PCA")
print("SKIP scaling: Decision Trees, Random Forests, XGBoost")


# ==============================================================
# SECTION 2: NORMALIZATION — MIN-MAX SCALING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: NORMALIZATION (MIN-MAX SCALER)")
print("=" * 60)

# Formula: X_scaled = (X - X_min) / (X_max - X_min)
# Result: ALL values become numbers between 0 and 1.
#   - The smallest value becomes 0
#   - The largest value becomes 1
#   - Everything else is in between

# When to use:
#   - When you need values strictly between 0 and 1
#   - For neural networks (especially image pixel values 0–255 → 0–1)
#   - WEAKNESS: outliers will distort the range (one huge value → everything squished near 0)

minmax_scaler = MinMaxScaler()   # Create the scaler object

# fit_transform:
#   .fit()      → learn the min and max of each column
#   .transform() → apply the formula to every value
df_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df_demo),   # Apply scaling
    columns=[col + "_minmax" for col in df_demo.columns]   # Rename columns
)

print("\nAfter Min-Max Scaling (all values 0–1):")
print(df_minmax.describe().round(4))


# ==============================================================
# SECTION 3: STANDARDIZATION — Z-SCORE SCALING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: STANDARDIZATION (STANDARD SCALER)")
print("=" * 60)

# Formula: X_scaled = (X - mean) / standard_deviation
# Result: data is centred at 0 with a standard deviation of 1.
#   - Mean becomes 0
#   - Most values fall between -3 and +3
#   - Outliers move to very large positive or negative numbers

# When to use:
#   - MOST common choice — works well in most situations
#   - Linear models, logistic regression, SVM
#   - Good when data follows a bell-curve (normal distribution)

standard_scaler = StandardScaler()
df_standard = pd.DataFrame(
    standard_scaler.fit_transform(df_demo),
    columns=[col + "_standard" for col in df_demo.columns]
)

print("\nAfter Standard Scaling (mean≈0, std≈1):")
print(df_standard.describe().round(4))


# ==============================================================
# SECTION 4: ROBUST SCALER — HANDLES OUTLIERS BETTER
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: ROBUST SCALER")
print("=" * 60)

# Formula: X_scaled = (X - median) / IQR
# Uses MEDIAN and IQR instead of mean and std.
# Because median and IQR are not affected by extreme outliers,
# this scaler is much more robust when your data has outliers.

# When to use:
#   - When data contains outliers that you can't or don't want to remove
#   - More stable than Standard Scaler for messy real-world data

robust_scaler = RobustScaler()
df_robust = pd.DataFrame(
    robust_scaler.fit_transform(df_demo),
    columns=[col + "_robust" for col in df_demo.columns]
)

print("\nAfter Robust Scaling (uses median & IQR):")
print(df_robust.describe().round(4))


# ==============================================================
# SECTION 5: VISUALISE — DISTRIBUTIONS BEFORE & AFTER
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: VISUALISING SCALING EFFECTS")
print("=" * 60)

# Compare distributions of 'income' before and after each scaler

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Income Distribution: Before & After Scaling", fontsize=14, fontweight="bold")

# Before scaling
axes[0, 0].hist(df_demo["income"], bins=15, color="salmon", edgecolor="black")
axes[0, 0].set_title("BEFORE Scaling")
axes[0, 0].set_xlabel("Income (£)")

# After Min-Max
axes[0, 1].hist(df_minmax["income_minmax"], bins=15, color="skyblue", edgecolor="black")
axes[0, 1].set_title("Min-Max (0 to 1)")
axes[0, 1].set_xlabel("Scaled Value")

# After Standard
axes[1, 0].hist(df_standard["income_standard"], bins=15, color="lightgreen", edgecolor="black")
axes[1, 0].set_title("Standard Scaler (mean=0, std=1)")
axes[1, 0].set_xlabel("Scaled Value")

# After Robust
axes[1, 1].hist(df_robust["income_robust"], bins=15, color="plum", edgecolor="black")
axes[1, 1].set_title("Robust Scaler (median-centred)")
axes[1, 1].set_xlabel("Scaled Value")

plt.tight_layout()
plt.savefig("charts/day08_scaling_comparison.png", dpi=100, bbox_inches="tight")
print("Scaling comparison chart saved → charts/day08_scaling_comparison.png")
plt.close()


# ==============================================================
# SECTION 6: WHICH SCALER TO CHOOSE?
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: CHOOSING THE RIGHT SCALER")
print("=" * 60)

comparison = [
    ("Min-Max Scaler",   "Bounded 0–1 output", "Neural networks, image data", "Sensitive to outliers"),
    ("Standard Scaler",  "Mean=0, Std=1",       "Most ML models (general)",   "Sensitive to outliers"),
    ("Robust Scaler",    "Median-centred, IQR",  "Data with outliers",          "Values may go beyond 0–1"),
]

print(f"\n{'Scaler':<20} {'Result':<25} {'Best For':<25} {'Weakness'}")
print("-" * 95)
for name, result, best, weakness in comparison:
    print(f"{name:<20} {result:<25} {best:<25} {weakness}")


# ==============================================================
# SECTION 7: SKLEARN PIPELINE — CHAIN STEPS TOGETHER
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: SKLEARN PIPELINE")
print("=" * 60)

# A Pipeline links preprocessing steps + a model in one clean object.
# This prevents "data leakage" — where test data accidentally influences training.
# (We'll use it more in model building — here's a taste!)

# For now, let's create a simple 2-step pipeline: imputer → scaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # Step 1: fill missing values
    ("scaler",  StandardScaler()),                   # Step 2: standardise
])

# Fit and transform in one command
df_pipeline_cols = ["age", "income", "purchase_amount"]
df_pipeline_result = pipeline.fit_transform(df[df_pipeline_cols])

print("\nPipeline result (impute → scale) — first 5 rows:")
print(pd.DataFrame(df_pipeline_result, columns=df_pipeline_cols).round(3).head())
print("\nThis is the recommended way — keeps steps tidy and prevents data leakage!")


# ==============================================================
# SECTION 8: APPLY SCALING AND SAVE
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 8: SAVE SCALED DATASET")
print("=" * 60)

# Apply Standard Scaler to all numeric columns
df_scaled = df.copy()

# Pick only numeric columns for scaling
# select_dtypes('number') picks float and int columns only
numeric_cols = df_scaled.select_dtypes(include="number").columns.tolist()

# Fit and transform
scaler = StandardScaler()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

output_path = "../data/day08_scaled.csv"
df_scaled.to_csv(output_path, index=False)
print(f"\nScaled dataset saved → {output_path}")
print(f"Shape: {df_scaled.shape}")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 8 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Scaling makes all features comparable in magnitude")
print("✓ Min-Max: squeezes everything to 0–1 (use for neural nets)")
print("✓ Standard: mean=0, std=1 (best general-purpose choice)")
print("✓ Robust: uses median/IQR — handles outliers better")
print("✓ Tree models (Random Forest, XGBoost) DON'T need scaling")
print("✓ Use sklearn Pipeline to chain imputation + scaling + model")
print("\nTomorrow: Full Exploratory Data Analysis (EDA)!")

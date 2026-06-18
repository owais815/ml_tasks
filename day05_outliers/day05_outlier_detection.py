# ==============================================================
# DAY 5 — OUTLIER DETECTION & TREATMENT
# Course: ML Fundamentals | Week 2
# ==============================================================

# An OUTLIER is a data point that is very different from the rest.
# Example: if most people earn £30,000–£80,000, someone earning
# £5,000,000 is an outlier — it could be a mistake or truly unusual.
# Outliers can BREAK ML models by pulling predictions in the wrong direction!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend so plots save to files

print("=" * 60)
print("DAY 5: OUTLIER DETECTION & TREATMENT")
print("=" * 60)

# Load the cleaned dataset from Day 4 (no missing values)
df = pd.read_csv("../data/day04_no_missing.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Create output folder for charts
import os
os.makedirs("charts", exist_ok=True)


# ==============================================================
# SECTION 1: UNDERSTANDING OUTLIERS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: WHAT ARE OUTLIERS?")
print("=" * 60)

# Let's look at purchase_amount — we know row 16 has a value of 50000!
print("\nStatistical summary of purchase_amount:")
print(df["purchase_amount"].describe().round(2))

# The max of 50000 vs a mean of ~500 is a BIG red flag!
print("\nStatistical summary of age:")
print(df["age"].describe().round(2))
# age has -5 and 150 which are impossible ages!


# ==============================================================
# SECTION 2: VISUAL DETECTION — BOXPLOTS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: VISUALISING OUTLIERS WITH BOXPLOTS")
print("=" * 60)

# A BOXPLOT shows:
#   - The box = where 50% of data lives (IQR)
#   - The line in the box = the median (middle value)
#   - The whiskers = acceptable range
#   - Dots beyond whiskers = OUTLIERS

fig, axes = plt.subplots(1, 3, figsize=(15, 5))   # Create 3 charts side by side
fig.suptitle("Boxplots for Outlier Detection", fontsize=14, fontweight="bold")

# Boxplot for purchase_amount
axes[0].boxplot(df["purchase_amount"])
axes[0].set_title("Purchase Amount")
axes[0].set_ylabel("£")

# Boxplot for age
axes[1].boxplot(df["age"])
axes[1].set_title("Age")
axes[1].set_ylabel("Years")

# Boxplot for income
axes[2].boxplot(df["income"])
axes[2].set_title("Income")
axes[2].set_ylabel("£")

plt.tight_layout()   # Prevents charts from overlapping
plt.savefig("charts/day05_boxplots.png", dpi=100, bbox_inches="tight")
print("Boxplot saved → charts/day05_boxplots.png")
plt.close()


# ==============================================================
# SECTION 3: METHOD 1 — Z-SCORE METHOD
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: Z-SCORE METHOD")
print("=" * 60)

# Z-SCORE measures how many STANDARD DEVIATIONS a value is from the mean.
# Formula: z = (value - mean) / standard_deviation
#
# Rule of thumb:
#   |z| > 3  →  the value is an outlier (3 standard deviations away)
#   Normal values: z is between -3 and +3

# Calculate z-score manually
col = "purchase_amount"
mean = df[col].mean()
std  = df[col].std()

# Apply the z-score formula to every row (pandas does this automatically)
df["purchase_zscore"] = (df[col] - mean) / std

# Flag outliers: any z-score beyond ±3 is suspicious
purchase_outliers_z = df[df["purchase_zscore"].abs() > 3]
print(f"\nOutliers in 'purchase_amount' using Z-score (|z| > 3):")
print(purchase_outliers_z[["customer_id", col, "purchase_zscore"]])

# Do the same for age
df["age_zscore"] = (df["age"] - df["age"].mean()) / df["age"].std()
age_outliers_z = df[df["age_zscore"].abs() > 3]
print(f"\nOutliers in 'age' using Z-score:")
print(age_outliers_z[["customer_id", "age", "age_zscore"]])


# ==============================================================
# SECTION 4: METHOD 2 — IQR (INTERQUARTILE RANGE) METHOD
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: IQR METHOD")
print("=" * 60)

# IQR = Interquartile Range = Q3 - Q1
# Q1 = 25th percentile (bottom quarter of data)
# Q3 = 75th percentile (top quarter of data)
# IQR = the "middle 50%" of your data
#
# Outlier boundaries:
#   Lower fence = Q1 - 1.5 × IQR   (anything below this is an outlier)
#   Upper fence = Q3 + 1.5 × IQR   (anything above this is an outlier)

def detect_outliers_iqr(series, label):
    """Detect outliers in a pandas Series using the IQR method."""
    Q1  = series.quantile(0.25)   # 25th percentile
    Q3  = series.quantile(0.75)   # 75th percentile
    IQR = Q3 - Q1                 # Interquartile range

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    print(f"\n  {label}:")
    print(f"    Q1 = {Q1:.2f},  Q3 = {Q3:.2f},  IQR = {IQR:.2f}")
    print(f"    Lower fence = {lower_fence:.2f}")
    print(f"    Upper fence = {upper_fence:.2f}")

    # Boolean mask: True where value is outside the fences
    outlier_mask = (series < lower_fence) | (series > upper_fence)
    return outlier_mask, lower_fence, upper_fence

print("IQR outlier detection:")
purchase_mask, p_low, p_high = detect_outliers_iqr(df["purchase_amount"], "Purchase Amount")
age_mask, a_low, a_high      = detect_outliers_iqr(df["age"], "Age")

print(f"\n  purchase_amount outliers: {purchase_mask.sum()}")
print(df[purchase_mask][["customer_id", "purchase_amount"]])

print(f"\n  age outliers: {age_mask.sum()}")
print(df[age_mask][["customer_id", "age"]])


# ==============================================================
# SECTION 5: TREATMENT — WHAT TO DO WITH OUTLIERS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: TREATING OUTLIERS")
print("=" * 60)

df_treated = df.copy()

# --- OPTION 1: REMOVE the outlier rows ---
# Only use when you're sure it's a data error.
df_removed = df_treated[~purchase_mask]   # ~ means NOT (keep non-outliers)
print(f"\nOption 1 — Remove outlier rows:")
print(f"  Before: {len(df_treated)} rows")
print(f"  After:  {len(df_removed)} rows")

# --- OPTION 2: CAP / WINSORIZE (clip to the fence values) ---
# Instead of removing, we "cap" extreme values at the fence.
# This keeps the row but limits the damage of the extreme value.

df_treated["purchase_amount_capped"] = df_treated["purchase_amount"].clip(
    lower=p_low,    # anything below lower_fence → set to lower_fence
    upper=p_high    # anything above upper_fence → set to upper_fence
)
print("\nOption 2 — Winsorize (cap) purchase_amount:")
print(df_treated[["customer_id","purchase_amount","purchase_amount_capped"]].sort_values("purchase_amount", ascending=False).head(5))

# Cap age outliers (impossible ages)
df_treated["age_capped"] = df_treated["age"].clip(lower=a_low, upper=a_high)
print("\nCapped age outliers:")
print(df_treated[df_treated["age"] != df_treated["age_capped"]][["customer_id","age","age_capped"]])

# --- OPTION 3: LOG TRANSFORM ---
# When data is very skewed (like purchase_amount), taking the log
# compresses the scale and reduces the impact of very large values.
# (1+ avoids log(0) which is undefined)
df_treated["purchase_log"] = np.log1p(df_treated["purchase_amount"])
print("\nOption 3 — Log transform of purchase_amount (first 5 rows):")
print(df_treated[["purchase_amount","purchase_log"]].head(5))


# ==============================================================
# SECTION 6: VISUALISING BEFORE & AFTER
# ==============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Before & After Outlier Treatment", fontsize=14, fontweight="bold")

# Before — purchase_amount (with outlier)
axes[0, 0].hist(df["purchase_amount"], bins=20, color="salmon", edgecolor="black")
axes[0, 0].set_title("purchase_amount — BEFORE (raw)")
axes[0, 0].set_xlabel("Amount (£)")

# After — purchase_amount capped
axes[0, 1].hist(df_treated["purchase_amount_capped"], bins=20, color="skyblue", edgecolor="black")
axes[0, 1].set_title("purchase_amount — AFTER (capped)")
axes[0, 1].set_xlabel("Amount (£)")

# Before — age (with outliers -5 and 150)
axes[1, 0].boxplot(df["age"])
axes[1, 0].set_title("age — BEFORE (raw)")
axes[1, 0].set_ylabel("Years")

# After — age capped
axes[1, 1].boxplot(df_treated["age_capped"])
axes[1, 1].set_title("age — AFTER (capped)")
axes[1, 1].set_ylabel("Years")

plt.tight_layout()
plt.savefig("charts/day05_before_after.png", dpi=100, bbox_inches="tight")
print("\nBefore/after charts saved → charts/day05_before_after.png")
plt.close()


# ==============================================================
# SECTION 7: SAVE TREATED DATASET
# ==============================================================

# Use capped versions for our cleaned dataset
df_final = df_treated.copy()
df_final["purchase_amount"] = df_final["purchase_amount_capped"]
df_final["age"]             = df_final["age_capped"]

# Remove the helper columns we no longer need
cols_to_drop = ["purchase_zscore","age_zscore","purchase_amount_capped","age_capped","purchase_log"]
df_final = df_final.drop(columns=cols_to_drop)

output_path = "../data/day05_no_outliers.csv"
df_final.to_csv(output_path, index=False)
print(f"\nTreated dataset saved → {output_path}")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 5 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Outliers are extreme values that can distort ML models")
print("✓ Detect visually with boxplots")
print("✓ Z-Score method: |z| > 3 = outlier")
print("✓ IQR method: below Q1-1.5*IQR or above Q3+1.5*IQR = outlier")
print("✓ Treatment: remove (if error), cap (Winsorize), or log-transform")
print("✓ Domain knowledge is ESSENTIAL — not every extreme value is bad!")
print("\nTomorrow: Feature Engineering Part 1!")

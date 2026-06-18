# ==============================================================
# DAY 4 — HANDLING MISSING VALUES
# Course: ML Fundamentals | Week 2
# ==============================================================

# Missing values are VERY common in real-world data.
# A cell with no value is called NaN (Not a Number) in pandas.
# Most ML algorithms CANNOT handle NaN — we must fix them first!

import pandas as pd
import numpy as np   # numpy = numerical Python (math operations)

print("=" * 60)
print("DAY 4: HANDLING MISSING VALUES")
print("=" * 60)

df = pd.read_csv("../data/raw_data.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")


# ==============================================================
# SECTION 1: WHY DATA GOES MISSING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: WHY DOES DATA GO MISSING?")
print("=" * 60)

# There are 3 types of missingness:

# MCAR — Missing Completely At Random
#   The missing values have NO pattern.
#   e.g., a survey respondent accidentally skipped a question.

# MAR — Missing At Random
#   Missing values have a pattern related to OTHER columns.
#   e.g., older customers less likely to fill in their income.

# MNAR — Missing Not At Random
#   The missing value itself causes it to be missing.
#   e.g., very rich people don't share their income.

print("MCAR → Missing randomly, no pattern (safest to impute)")
print("MAR  → Missing due to another variable (impute carefully)")
print("MNAR → Missing because of the value itself (hardest case)")


# ==============================================================
# SECTION 2: DETECTING MISSING VALUES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: DETECTING MISSING VALUES")
print("=" * 60)

# .isnull() returns True where value is missing, False where it exists
# .sum() then counts how many True values (missing) per column

missing_counts = df.isnull().sum()
print("\nMissing values per column:")
print(missing_counts[missing_counts > 0])   # Only show columns WITH missing values

# Calculate percentage of missing values per column
missing_pct = (df.isnull().sum() / len(df)) * 100
print("\nMissing values as percentage (%):")
print(missing_pct[missing_pct > 0].round(2))

# Total missing values in the entire dataset
total_missing = df.isnull().sum().sum()   # .sum() twice: once per column, once total
print(f"\nTotal missing cells: {total_missing}")

# .info() also shows how many NON-null values each column has
print("\nDataFrame info (shows non-null counts):")
df.info()


# ==============================================================
# SECTION 3: STRATEGY 1 — DROPPING MISSING VALUES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: DROPPING ROWS OR COLUMNS WITH MISSING VALUES")
print("=" * 60)

# WHEN TO DROP:
#   - Only if the missing percentage is VERY small (e.g., < 2%)
#   - Or if the column is almost entirely missing (> 70% missing)
#   - WARNING: Dropping rows removes data — use sparingly!

# Drop rows where ANY column has a missing value
df_dropped_rows = df.dropna()
print(f"\nOriginal rows: {len(df)}")
print(f"After dropping any-NaN rows: {len(df_dropped_rows)}")
print(f"Rows lost: {len(df) - len(df_dropped_rows)}")

# Drop rows where a SPECIFIC column has a missing value
df_drop_education = df.dropna(subset=["education"])
print(f"\nAfter dropping rows with missing 'education': {len(df_drop_education)} rows")

# Drop COLUMNS where more than 50% of values are missing
# (Not relevant here, but here's how you'd do it)
threshold = len(df) * 0.5   # 50% of total rows
df_drop_cols = df.dropna(axis=1, thresh=int(threshold))
print(f"\nColumns after dropping >50% missing cols: {df_drop_cols.shape[1]}")


# ==============================================================
# SECTION 4: STRATEGY 2 — MEAN / MEDIAN / MODE IMPUTATION
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: IMPUTATION — FILLING IN MISSING VALUES")
print("=" * 60)

# "Imputation" means replacing NaN with an estimated value.
# NEVER modify the original df — always work on a copy!
df_imputed = df.copy()   # .copy() makes a separate copy so original stays safe

# --- Mean Imputation ---
# Fill with the AVERAGE value of the column.
# Best for: normally distributed data (no extreme outliers).

income_mean = df_imputed["income"].mean()
print(f"\nIncome — Mean: £{income_mean:,.0f}")
df_imputed["income_mean_filled"] = df_imputed["income"].fillna(income_mean)
print("income filled with MEAN (first 20 rows showing missing rows):")
print(df_imputed[df_imputed["income"].isnull()][["customer_id","income","income_mean_filled"]])

# --- Median Imputation ---
# Fill with the MIDDLE value of the column.
# Best for: skewed data or data with outliers (more robust than mean).

income_median = df_imputed["income"].median()
print(f"\nIncome — Median: £{income_median:,.0f}")
df_imputed["income_median_filled"] = df_imputed["income"].fillna(income_median)

# --- Mode Imputation ---
# Fill with the MOST FREQUENT value.
# Best for: CATEGORICAL (text) columns.

education_mode = df_imputed["education"].mode()[0]   # [0] gets the first/most common value
print(f"\nEducation — Mode (most common): '{education_mode}'")
df_imputed["education_mode_filled"] = df_imputed["education"].fillna(education_mode)
print("Education filled with mode (showing previously missing rows):")
print(df_imputed[df_imputed["education"].isnull()][["customer_id","education","education_mode_filled"]])

# Fill customer_rating with median (ratings can be skewed)
rating_median = df_imputed["customer_rating"].median()
df_imputed["customer_rating"] = df_imputed["customer_rating"].fillna(rating_median)
print(f"\ncustomer_rating filled with median: {rating_median}")


# ==============================================================
# SECTION 5: STRATEGY 3 — FORWARD FILL & BACK FILL
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: FORWARD FILL & BACK FILL")
print("=" * 60)

# These strategies work well for TIME SERIES data
# where values change gradually over time.

# Forward fill (ffill): fills NaN with the PREVIOUS row's value
# Back fill  (bfill): fills NaN with the NEXT row's value

df_time = df.copy()

print("\nBefore forward fill — income column (rows with NaN):")
print(df_time[df_time["income"].isnull()][["customer_id","income"]].head(3))

df_time["income_ffill"] = df_time["income"].ffill()
print("\nAfter forward fill:")
print(df_time[df["income"].isnull()][["customer_id","income","income_ffill"]].head(3))

df_time["income_bfill"] = df_time["income"].bfill()
print("\nAfter back fill:")
print(df_time[df["income"].isnull()][["customer_id","income","income_bfill"]].head(3))


# ==============================================================
# SECTION 6: COMPARING IMPUTATION STRATEGIES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: COMPARING IMPUTATION STRATEGIES")
print("=" * 60)

# Let's compare how each strategy affects the statistical summary
print("\nOriginal income stats:")
print(df["income"].describe().round(2))

print("\nIncome — filled with MEAN:")
print(df_imputed["income_mean_filled"].describe().round(2))

print("\nIncome — filled with MEDIAN:")
print(df_imputed["income_median_filled"].describe().round(2))

# KEY INSIGHT:
# Mean imputation keeps the mean the same but reduces variance (spread).
# Median imputation is safer when there are outliers.


# ==============================================================
# SECTION 7: FINAL CLEANED DATASET
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: SAVING THE CLEANED DATASET")
print("=" * 60)

# Apply the best strategy to each column and save
df_clean = df.copy()

# income → use median (safer, data may be skewed)
df_clean["income"] = df_clean["income"].fillna(df_clean["income"].median())

# education → use mode (categorical column)
df_clean["education"] = df_clean["education"].fillna(df_clean["education"].mode()[0])

# customer_rating → use median
df_clean["customer_rating"] = df_clean["customer_rating"].fillna(df_clean["customer_rating"].median())

# Confirm no more missing values
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Save the cleaned data
output_path = "../data/day04_no_missing.csv"
df_clean.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 4 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ NaN = missing value — most ML models cannot handle them")
print("✓ Detect: df.isnull().sum()")
print("✓ Drop rows: df.dropna() — only if very few are missing")
print("✓ Mean fill: good for normal distributions")
print("✓ Median fill: better when there are outliers")
print("✓ Mode fill: use for categorical (text) columns")
print("✓ Forward/back fill: best for time series")
print("\nTomorrow: Outlier Detection & Treatment!")

# ==============================================================
# DAY 7 — FEATURE ENGINEERING PART 2: ENCODING CATEGORICAL VARIABLES
# Course: ML Fundamentals | Week 3
# ==============================================================

# ML algorithms work with NUMBERS, not text!
# Columns like 'gender' (Male/Female) or 'city' (London/Leeds) are text.
# We must convert them to numbers — this is called ENCODING.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

print("=" * 60)
print("DAY 7: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

df = pd.read_csv("../data/day06_engineered.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Which columns are categorical (text)?
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("\nCategorical (text) columns:", cat_cols)


# ==============================================================
# SECTION 1: TYPES OF CATEGORICAL DATA
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: NOMINAL vs ORDINAL CATEGORICAL DATA")
print("=" * 60)

# NOMINAL: Categories with NO natural order
#   - Examples: city (London, Leeds, Glasgow), gender (Male, Female)
#   - London is NOT "greater" than Leeds — they're just different.
#   - Use ONE-HOT ENCODING for these.

# ORDINAL: Categories that have a NATURAL ORDER
#   - Examples: education (High School < Bachelor < Master < PhD)
#   - PhD IS greater/more than High School — order matters!
#   - Use LABEL/ORDINAL ENCODING for these.

print("NOMINAL (no order) → One-Hot Encoding")
print("  Examples: city, gender, product_category")
print("\nORDINAL (has order) → Label / Ordinal Encoding")
print("  Examples: education (High School → Bachelor → Master → PhD)")

print("\nValue counts of 'education':")
print(df["education"].value_counts())

print("\nValue counts of 'city':")
print(df["city"].value_counts())


# ==============================================================
# SECTION 2: LABEL ENCODING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: LABEL ENCODING")
print("=" * 60)

# Label Encoding assigns a UNIQUE INTEGER to each category.
# It's fast and simple, but has a PROBLEM:
# The model may think "PhD (3) > Master (2) > Bachelor (1)" — like numbers!
# This is OK for ordinal data, but WRONG for nominal data.

df_encoded = df.copy()   # Always work on a copy!

# Encode 'gender' with LabelEncoder
le_gender = LabelEncoder()
# fit_transform does two things:
#   .fit()      → learn the categories
#   .transform() → convert them to numbers
df_encoded["gender_label"] = le_gender.fit_transform(df_encoded["gender"])

print("\nLabel Encoding — 'gender':")
print("  Mapping:", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
print(df_encoded[["gender","gender_label"]].drop_duplicates().sort_values("gender_label"))


# ==============================================================
# SECTION 3: ORDINAL ENCODING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: ORDINAL ENCODING (for ordered categories)")
print("=" * 60)

# Ordinal Encoding lets us SPECIFY the order ourselves.
# This is perfect for 'education' — we know the order!

education_order = [["High School", "Bachelor", "Master", "PhD"]]
# Note: categories are in a list-of-lists because OrdinalEncoder handles multiple columns

oe = OrdinalEncoder(categories=education_order)

# OrdinalEncoder expects a 2D input: df[["column"]] not df["column"]
df_encoded["education_ordinal"] = oe.fit_transform(df_encoded[["education"]])

print("\nOrdinal Encoding — 'education':")
edu_map = df_encoded[["education","education_ordinal"]].drop_duplicates().sort_values("education_ordinal")
print(edu_map)
print("\nHigh School=0, Bachelor=1, Master=2, PhD=3 — the ORDER is preserved!")


# ==============================================================
# SECTION 4: ONE-HOT ENCODING
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: ONE-HOT ENCODING")
print("=" * 60)

# One-Hot Encoding creates a NEW BINARY COLUMN for each category.
# If city has 5 unique values → 5 new columns, each 0 or 1.
# Only ONE column is 1 for each row (that's why it's called "one-hot").

# This avoids the false "ordering" problem of label encoding.

print("\nUnique cities:", df_encoded["city"].unique())
print(f"Shape BEFORE one-hot encoding: {df_encoded.shape}")

# pd.get_dummies() is the simplest way to one-hot encode
# drop_first=True drops one column to avoid the "dummy variable trap"
# (if we know 4 cities, the 5th is implied — one column is redundant)
city_dummies = pd.get_dummies(df_encoded["city"], prefix="city", drop_first=False)

print("\nOne-hot encoded 'city' (first 5 rows):")
print(city_dummies.head(5))

# Similarly for product_category
cat_dummies = pd.get_dummies(df_encoded["product_category"], prefix="cat", drop_first=False)

# Join the new columns to our DataFrame
df_ohe = pd.concat([df_encoded, city_dummies, cat_dummies], axis=1)
# axis=1 means join column-wise (side by side)

print(f"\nShape AFTER one-hot encoding: {df_ohe.shape}")
print("New one-hot columns added:", city_dummies.columns.tolist() + cat_dummies.columns.tolist())


# ==============================================================
# SECTION 5: DUMMY VARIABLE TRAP
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: THE DUMMY VARIABLE TRAP")
print("=" * 60)

# When you one-hot encode N categories, you only NEED N-1 columns.
# The last one is redundant — it can always be derived from the others.
# Example: if is_Male=0 and is_Female=0 → must be "Other"
# This redundancy can confuse linear ML models.

# Solution: use drop_first=True in get_dummies()
gender_dummies_trap     = pd.get_dummies(df_encoded["gender"], prefix="gender")
gender_dummies_no_trap  = pd.get_dummies(df_encoded["gender"], prefix="gender", drop_first=True)

print(f"\nGender one-hot WITH trap:    {gender_dummies_trap.shape[1]} columns: {gender_dummies_trap.columns.tolist()}")
print(f"Gender one-hot WITHOUT trap:  {gender_dummies_no_trap.shape[1]} columns: {gender_dummies_no_trap.columns.tolist()}")
print("\nWith drop_first=True, one column is removed (it's implied by the others).")


# ==============================================================
# SECTION 6: COMPARING SHAPES BEFORE & AFTER
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: BEFORE vs AFTER SUMMARY")
print("=" * 60)

print(f"\nOriginal DataFrame shape:         {df.shape}")
print(f"After all encoding (city+cat):    {df_ohe.shape}")
print(f"\nAdded {df_ohe.shape[1] - df.shape[1]} new columns from encoding")


# ==============================================================
# SECTION 7: FINAL ENCODED DATASET (READY FOR ML)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: SAVING THE ENCODED DATASET")
print("=" * 60)

# Drop the original text columns (model can't use them)
# Keep the ordinal encoded education, one-hot gender, city, category
df_final = df.copy()

# Ordinal encode education
df_final["education_ordinal"] = oe.fit_transform(df_final[["education"]])

# One-hot encode gender (drop_first to avoid dummy trap)
gender_oh = pd.get_dummies(df_final["gender"], prefix="gender", drop_first=True)

# One-hot encode city
city_oh = pd.get_dummies(df_final["city"], prefix="city", drop_first=True)

# One-hot encode product_category
cat_oh = pd.get_dummies(df_final["product_category"], prefix="cat", drop_first=True)

# Join all new columns
df_final = pd.concat([df_final, gender_oh, city_oh, cat_oh], axis=1)

# Drop original text columns now that we have numeric versions
text_cols_to_drop = ["gender", "city", "product_category", "education", "income_bracket"]
df_final = df_final.drop(columns=text_cols_to_drop, errors="ignore")

output_path = "../data/day07_encoded.csv"
df_final.to_csv(output_path, index=False)
print(f"\nEncoded dataset saved → {output_path}")
print(f"Final shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 7 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ ML needs numbers — text columns must be encoded first")
print("✓ Nominal (no order): use One-Hot Encoding (pd.get_dummies())")
print("✓ Ordinal (has order): use Ordinal Encoding (specify the order!)")
print("✓ Label Encoding: assigns integers — use only for ordinal or tree models")
print("✓ Dummy Variable Trap: use drop_first=True to remove one redundant column")
print("✓ One-hot encoding increases the number of columns — be careful with many categories!")
print("\nTomorrow: Feature Scaling!")

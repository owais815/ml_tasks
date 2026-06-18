# ==============================================================
# DAY 3 — PANDAS DEEP DIVE
# Course: ML Fundamentals | Week 2
# ==============================================================

# Pandas is the most important library for working with data in Python.
# Think of it as Excel, but much more powerful and inside Python!

import pandas as pd   # Import pandas and call it 'pd'

print("=" * 60)
print("DAY 3: PANDAS DEEP DIVE")
print("=" * 60)

# Load our dataset
df = pd.read_csv("../data/raw_data.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")


# ==============================================================
# SECTION 1: SERIES vs DATAFRAME
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: SERIES vs DATAFRAME")
print("=" * 60)

# A SERIES is a single column (1D — one dimension, like a list with labels)
age_series = df["age"]
print("\nSeries — just the 'age' column:")
print(age_series.head(5))   # Show first 5 values
print("Type:", type(age_series))   # Confirms it's a Series

# A DATAFRAME is a full table (2D — rows AND columns)
# Our 'df' is already a DataFrame
print("\nDataFrame — the full table:")
print(df.head(3))
print("Type:", type(df))


# ==============================================================
# SECTION 2: SELECTING COLUMNS & ROWS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: SELECTING COLUMNS & ROWS")
print("=" * 60)

# --- Selecting ONE column ---
# Use df["column_name"] to get a single column (returns a Series)
print("\nSingle column — purchase_amount:")
print(df["purchase_amount"].head(5))

# --- Selecting MULTIPLE columns ---
# Use df[["col1", "col2"]] — note the double square brackets [[...]]
print("\nMultiple columns — age, gender, purchase_amount:")
print(df[["age", "gender", "purchase_amount"]].head(5))


# --- .loc[ ] — Select by LABEL (row label AND column name) ---
# loc[row_label, column_name]
# df.loc[0]  → row with INDEX label 0
# df.loc[0:4, "age":"city"]  → rows 0-4, columns from age to city

print("\n.loc[] — select rows 0-2, columns age to city:")
print(df.loc[0:2, "age":"city"])


# --- .iloc[ ] — Select by POSITION (integer position, like a grid) ---
# iloc[row_number, column_number]  — counting starts at 0!

print("\n.iloc[] — select rows 0-2, columns 0-4 (by position):")
print(df.iloc[0:3, 0:5])


# ==============================================================
# SECTION 3: FILTERING DATA (BOOLEAN INDEXING)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: FILTERING DATA")
print("=" * 60)

# Boolean indexing = filter rows based on a condition.
# The condition returns True/False for each row.
# Only rows where the condition is True are kept.

# Filter: customers from London only
london_customers = df[df["city"] == "London"]
print(f"\nCustomers from London: {len(london_customers)} rows")
print(london_customers[["customer_id", "age", "city", "purchase_amount"]].head())

# Filter: customers who spent more than £200
big_spenders = df[df["purchase_amount"] > 200]
print(f"\nCustomers spending > £200: {len(big_spenders)} rows")

# Filter: MULTIPLE conditions with & (AND) or | (OR)
# IMPORTANT: wrap each condition in parentheses ( )
female_london = df[(df["gender"] == "Female") & (df["city"] == "London")]
print(f"\nFemale customers in London: {len(female_london)} rows")
print(female_london[["customer_id", "age", "gender", "city"]].head())

# Filter: members OR high income
members_or_rich = df[(df["is_member"] == 1) | (df["income"] > 70000)]
print(f"\nMembers or high income: {len(members_or_rich)} rows")


# ==============================================================
# SECTION 4: SORTING DATA
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: SORTING DATA")
print("=" * 60)

# .sort_values() sorts the DataFrame by a column
# ascending=True  → smallest first (A→Z, 0→9)
# ascending=False → largest first (Z→A, 9→0)

print("\nTop 5 customers by purchase_amount (highest first):")
top_spenders = df.sort_values("purchase_amount", ascending=False)
print(top_spenders[["customer_id", "purchase_amount", "product_category"]].head(5))

print("\nYoungest 5 customers:")
youngest = df.sort_values("age", ascending=True)
print(youngest[["customer_id", "age", "city"]].head(5))

# Sort by MULTIPLE columns
print("\nSorted by city (A→Z) then purchase_amount (high→low):")
multi_sort = df.sort_values(["city", "purchase_amount"], ascending=[True, False])
print(multi_sort[["city", "purchase_amount", "gender"]].head(8))


# ==============================================================
# SECTION 5: GROUPBY — SUMMARISING BY CATEGORY
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: GROUPBY — GROUP AND SUMMARISE")
print("=" * 60)

# groupby() splits data into groups, then applies a function to each group.
# Like a "pivot table" in Excel.

# Average purchase amount PER city
city_avg = df.groupby("city")["purchase_amount"].mean()
print("\nAverage purchase amount by city:")
print(city_avg.round(2))   # round(2) limits to 2 decimal places

# Total purchases per product category
category_total = df.groupby("product_category")["purchase_amount"].sum()
print("\nTotal purchase amount by product category:")
print(category_total.round(2))

# Multiple aggregations at once using .agg()
print("\nSummary by gender (mean age, total spend, count):")
gender_summary = df.groupby("gender").agg(
    avg_age=("age", "mean"),              # mean age per gender
    total_spend=("purchase_amount", "sum"),# total spending
    num_customers=("customer_id", "count") # how many customers
)
print(gender_summary.round(2))


# ==============================================================
# SECTION 6: ADDING & MODIFYING COLUMNS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: ADDING NEW COLUMNS")
print("=" * 60)

# You can add a new column by assigning to df["new_column"]
# The operation is applied to every row automatically!

# Calculate discount amount (how much money was saved)
df["discount_amount"] = df["purchase_amount"] * df["discount_pct"] / 100
print("\nNew column 'discount_amount' (first 5 rows):")
print(df[["purchase_amount", "discount_pct", "discount_amount"]].head())

# Calculate final price after discount
df["final_price"] = df["purchase_amount"] - df["discount_amount"]
print("\nNew column 'final_price' (first 5 rows):")
print(df[["purchase_amount", "discount_amount", "final_price"]].head())


# ==============================================================
# SECTION 7: EXPORTING TO CSV
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: SAVING / EXPORTING DATA")
print("=" * 60)

output_path = "../data/day03_pandas_output.csv"

# index=False means don't save the row numbers (0, 1, 2...) as a column
df.to_csv(output_path, index=False)
print(f"\nCleaned DataFrame saved to: {output_path}")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 3 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Series = one column | DataFrame = full table")
print("✓ Select columns: df['col'] or df[['col1','col2']]")
print("✓ Select rows:    df.loc[label] or df.iloc[position]")
print("✓ Filter rows:    df[df['col'] > value]  (boolean indexing)")
print("✓ Sort data:      df.sort_values('col')")
print("✓ Group & summarise: df.groupby('col').agg(...)")
print("✓ Add new columns: df['new'] = df['a'] + df['b']")
print("✓ Save to CSV:    df.to_csv('path.csv', index=False)")
print("\nTomorrow: Handling Missing Values!")

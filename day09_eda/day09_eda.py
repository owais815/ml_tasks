# ==============================================================
# DAY 9 — EXPLORATORY DATA ANALYSIS (EDA)
# Course: ML Fundamentals | Week 3
# ==============================================================

# EDA = Exploring your data BEFORE building a model.
# The goal is to UNDERSTAND the data:
#   - What are the distributions?
#   - Which features are related?
#   - Are there patterns or anomalies?
#   - What story does the data tell?

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns   # Seaborn = beautiful statistical visualizations
import os

print("=" * 60)
print("DAY 9: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Load the clean dataset (before scaling — original values are easier to interpret!)
df = pd.read_csv("../data/day06_engineered.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

os.makedirs("charts", exist_ok=True)

# Set a visual style for all seaborn plots
sns.set_theme(style="whitegrid", palette="Set2")


# ==============================================================
# SECTION 1: EDA GOALS & QUICK OVERVIEW
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: QUICK DATASET OVERVIEW")
print("=" * 60)

print("\n.shape — rows × columns:")
print(df.shape)

print("\n.dtypes — data type of each column:")
print(df.dtypes)

print("\n.describe() — statistics for numeric columns:")
print(df.describe().round(2))

print("\n.info() — overview with null counts:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# ==============================================================
# SECTION 2: UNIVARIATE ANALYSIS — ONE COLUMN AT A TIME
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: UNIVARIATE ANALYSIS")
print("=" * 60)

# UNIVARIATE = analysing ONE variable at a time.
# We look at its distribution — how are values spread out?

# --- Numeric columns: histograms & boxplots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Univariate Analysis — Numeric Columns", fontsize=14, fontweight="bold")

numeric_cols_to_plot = ["age", "income", "purchase_amount", "customer_rating",
                        "num_purchases_last_year", "discount_pct"]

for i, col in enumerate(numeric_cols_to_plot):
    row = i // 3   # Which row of the grid
    col_idx = i % 3   # Which column of the grid
    ax = axes[row, col_idx]
    ax.hist(df[col], bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    # Draw a vertical line at the mean
    ax.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("charts/day09_univariate_numeric.png", dpi=100, bbox_inches="tight")
print("Univariate numeric plots saved → charts/day09_univariate_numeric.png")
plt.close()

# --- Categorical columns: value_counts bar charts ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Univariate Analysis — Categorical Columns", fontsize=14, fontweight="bold")

cat_cols = ["gender", "city", "product_category"]
for i, col in enumerate(cat_cols):
    counts = df[col].value_counts()
    axes[i].bar(counts.index, counts.values, color="coral", edgecolor="black")
    axes[i].set_title(f"'{col}' distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("charts/day09_univariate_categorical.png", dpi=100, bbox_inches="tight")
print("Univariate categorical plots saved → charts/day09_univariate_categorical.png")
plt.close()

# Print value counts in terminal too
for col in cat_cols:
    print(f"\n'{col}' value counts:")
    print(df[col].value_counts())


# ==============================================================
# SECTION 3: BIVARIATE ANALYSIS — TWO COLUMNS AT A TIME
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: BIVARIATE ANALYSIS")
print("=" * 60)

# BIVARIATE = analysing TWO variables to see if they're related.

# --- Scatter plot: income vs purchase_amount ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Bivariate Analysis — Scatter Plots", fontsize=14, fontweight="bold")

axes[0].scatter(df["income"], df["purchase_amount"], alpha=0.6, color="teal", edgecolors="black", linewidths=0.5)
axes[0].set_xlabel("Income (£)")
axes[0].set_ylabel("Purchase Amount (£)")
axes[0].set_title("Income vs Purchase Amount")

axes[1].scatter(df["age"], df["purchase_amount"], alpha=0.6, color="orange", edgecolors="black", linewidths=0.5)
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Purchase Amount (£)")
axes[1].set_title("Age vs Purchase Amount")

axes[2].scatter(df["num_purchases_last_year"], df["purchase_amount"], alpha=0.6, color="purple", edgecolors="black", linewidths=0.5)
axes[2].set_xlabel("Num Purchases Last Year")
axes[2].set_ylabel("Purchase Amount (£)")
axes[2].set_title("Num Purchases vs Amount")

plt.tight_layout()
plt.savefig("charts/day09_bivariate_scatter.png", dpi=100, bbox_inches="tight")
print("Bivariate scatter plots saved → charts/day09_bivariate_scatter.png")
plt.close()

# --- Cross-tab: gender vs city ---
cross = pd.crosstab(df["gender"], df["city"])
print("\nCross-tab: Gender vs City:")
print(cross)

# --- GroupBy: average purchase by city ---
city_avg = df.groupby("city")["purchase_amount"].mean().sort_values(ascending=False)
print("\nAverage purchase amount by city:")
print(city_avg.round(2))


# ==============================================================
# SECTION 4: MULTIVARIATE ANALYSIS — MANY COLUMNS AT ONCE
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: MULTIVARIATE ANALYSIS")
print("=" * 60)

# MULTIVARIATE = looking at 3+ variables together.

# --- Correlation Heatmap ---
# Heatmap shows correlation between ALL numeric columns at once.
# Red = positive correlation, Blue = negative, White = no correlation

numeric_cols = ["age", "income", "purchase_amount", "discount_pct",
                "customer_rating", "num_purchases_last_year",
                "is_member", "returns_count", "final_price"]

corr_matrix = df[numeric_cols].corr()   # Compute correlation matrix

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,          # Show the correlation number inside each cell
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Color scheme: red (positive) → blue (negative)
    center=0,            # Centre color at 0 (no correlation)
    linewidths=0.5,      # Add grid lines
    ax=ax
)
ax.set_title("Correlation Heatmap — All Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/day09_correlation_heatmap.png", dpi=100, bbox_inches="tight")
print("\nCorrelation heatmap saved → charts/day09_correlation_heatmap.png")
plt.close()

# Print strongest correlations with purchase_amount
corr_with_target = corr_matrix["purchase_amount"].drop("purchase_amount").sort_values(key=abs, ascending=False)
print("\nTop correlations with 'purchase_amount':")
print(corr_with_target.round(3))

# --- Pairplot (scatter matrix) ---
# Shows scatter plots between ALL pairs of selected columns
pairplot_cols = ["age", "income", "purchase_amount", "customer_rating", "num_purchases_last_year"]
pair_df = df[pairplot_cols + ["gender"]].copy()

pair_fig = sns.pairplot(pair_df, hue="gender", diag_kind="hist", plot_kws={"alpha": 0.5})
pair_fig.fig.suptitle("Pairplot — Colored by Gender", y=1.02, fontsize=13)
pair_fig.savefig("charts/day09_pairplot.png", dpi=80, bbox_inches="tight")
print("Pairplot saved → charts/day09_pairplot.png")
plt.close()


# ==============================================================
# SECTION 5: IDENTIFYING PATTERNS & ANOMALIES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: PATTERNS & ANOMALIES")
print("=" * 60)

# Check purchase amount by product category
cat_stats = df.groupby("product_category")["purchase_amount"].agg(["mean","median","count"])
print("\nPurchase amount by product category:")
print(cat_stats.round(2))

# Check if members spend more than non-members
member_vs_non = df.groupby("is_member")["purchase_amount"].mean()
print("\nAverage spend — Members vs Non-Members:")
print(member_vs_non.round(2))
print("Members spend more!" if member_vs_non[1] > member_vs_non[0] else "Non-members spend more!")

# Distribution of purchases by month
monthly = df.groupby("purchase_month")["purchase_amount"].sum()
print("\nTotal spending by month:")
print(monthly.round(2))


# ==============================================================
# SECTION 6: EDA OBSERVATIONS (TEMPLATE)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: EDA OBSERVATIONS (YOUR REPORT)")
print("=" * 60)

print("""
EDA Observations on Customer Shopping Dataset:
-----------------------------------------------
1. PURCHASE AMOUNT is right-skewed — most customers spend small amounts,
   but a few spend very large amounts (Electronics category).

2. INCOME and PURCHASE AMOUNT show a moderate positive correlation (~0.4).
   Higher-income customers tend to spend more per transaction.

3. ELECTRONICS is the highest-spending category on average,
   while FOOD has the smallest average transaction value.

4. MEMBERS spend significantly more on average than non-members.
   This suggests the membership programme attracts higher-value customers.

5. AGE shows no strong linear relationship with purchase amount,
   but customers aged 40–60 (PhD holders) tend to make larger purchases.
""")


# ==============================================================
# SUMMARY
# ==============================================================

print("=" * 60)
print("DAY 9 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ EDA = exploring data BEFORE modelling to understand it")
print("✓ Univariate: one column — histograms, value_counts")
print("✓ Bivariate:  two columns — scatter plots, cross-tabs, groupby")
print("✓ Multivariate: all columns — correlation heatmap, pairplot")
print("✓ Always look for: distributions, outliers, correlations, patterns")
print("\nTomorrow: Matplotlib & Seaborn Visualisation Dashboard!")

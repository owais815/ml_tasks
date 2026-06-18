# ==============================================================
# DAY 10 — DATA VISUALIZATION (MATPLOTLIB & SEABORN)
# Course: ML Fundamentals | Week 3
# ==============================================================

# Visualisation = turning data into charts, graphs, and plots.
# "A picture is worth a thousand rows."
# Good charts reveal patterns that tables never could.
#
# MATPLOTLIB — the foundation library for all Python plotting
# SEABORN    — built ON TOP of matplotlib, adds beautiful statistical charts

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive mode — saves charts to files
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("DAY 10: DATA VISUALIZATION")
print("=" * 60)

df = pd.read_csv("../data/day06_engineered.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")   # Global seaborn style


# ==============================================================
# SECTION 1: MATPLOTLIB FUNDAMENTALS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: MATPLOTLIB BASICS")
print("=" * 60)

# MATPLOTLIB ANATOMY:
#   Figure = the whole canvas (the "paper")
#   Axes   = one individual chart (one "plot area" on the paper)
#   You can have multiple Axes on one Figure (subplots)

# plt.figure(figsize=(width, height)) — width & height in inches
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)   # 1 row, 1 column, chart #1

x = [1, 2, 3, 4, 5]
y = [10, 25, 15, 40, 30]

ax.plot(x, y, color="steelblue", linewidth=2, marker="o", label="Sales")
ax.set_title("Simple Line Plot", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Sales (£)")
ax.legend()   # Show the legend (the label= we set above)
ax.grid(True, alpha=0.3)   # alpha controls transparency (0=invisible, 1=solid)

plt.tight_layout()
plt.savefig("charts/day10_01_line_plot.png", dpi=100)
print("Line plot saved → charts/day10_01_line_plot.png")
plt.close()


# ==============================================================
# SECTION 2: BAR CHART — COMPARING CATEGORIES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: BAR CHART")
print("=" * 60)

# Best for: comparing values across discrete categories

avg_by_city = df.groupby("city")["purchase_amount"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(avg_by_city.index, avg_by_city.values, color="coral", edgecolor="black", width=0.6)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
            f"£{height:.0f}", ha="center", va="bottom", fontsize=10)

ax.set_title("Average Purchase Amount by City", fontsize=13, fontweight="bold")
ax.set_xlabel("City")
ax.set_ylabel("Average Purchase (£)")
ax.set_ylim(0, avg_by_city.max() * 1.2)   # Give space for labels

plt.tight_layout()
plt.savefig("charts/day10_02_bar_chart.png", dpi=100)
print("Bar chart saved → charts/day10_02_bar_chart.png")
plt.close()


# ==============================================================
# SECTION 3: HISTOGRAM — DISTRIBUTION OF A SINGLE COLUMN
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: HISTOGRAM")
print("=" * 60)

# Best for: seeing how a numeric variable is distributed

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["purchase_amount"], bins=20, color="steelblue", edgecolor="white", alpha=0.8)

# Add vertical lines for mean and median
ax.axvline(df["purchase_amount"].mean(),   color="red",    linestyle="--", linewidth=2, label=f"Mean: £{df['purchase_amount'].mean():.0f}")
ax.axvline(df["purchase_amount"].median(), color="orange", linestyle="--", linewidth=2, label=f"Median: £{df['purchase_amount'].median():.0f}")

ax.set_title("Distribution of Purchase Amount", fontsize=13, fontweight="bold")
ax.set_xlabel("Purchase Amount (£)")
ax.set_ylabel("Number of Customers")
ax.legend()

plt.tight_layout()
plt.savefig("charts/day10_03_histogram.png", dpi=100)
print("Histogram saved → charts/day10_03_histogram.png")
plt.close()


# ==============================================================
# SECTION 4: SCATTER PLOT — RELATIONSHIP BETWEEN TWO VARIABLES
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: SCATTER PLOT")
print("=" * 60)

# Best for: seeing if two numeric variables are related

fig, ax = plt.subplots(figsize=(8, 6))

# Color points by gender for extra dimension
colors = {"Male": "steelblue", "Female": "coral", "Other": "green"}
for gender, group in df.groupby("gender"):
    ax.scatter(group["income"], group["purchase_amount"],
               label=gender, color=colors[gender],
               alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
    # s=60 controls dot size

ax.set_title("Income vs Purchase Amount (by Gender)", fontsize=13, fontweight="bold")
ax.set_xlabel("Income (£)")
ax.set_ylabel("Purchase Amount (£)")
ax.legend(title="Gender")

plt.tight_layout()
plt.savefig("charts/day10_04_scatter_plot.png", dpi=100)
print("Scatter plot saved → charts/day10_04_scatter_plot.png")
plt.close()


# ==============================================================
# SECTION 5: SEABORN — VIOLIN PLOT & BOX PLOT
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: SEABORN VIOLIN & BOX PLOTS")
print("=" * 60)

# BOX PLOT: shows median, IQR, and outliers
# VIOLIN PLOT: shows the full distribution shape (like a box + density curve)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Purchase Amount Distribution by Product Category", fontsize=13, fontweight="bold")

# Box plot
sns.boxplot(data=df, x="product_category", y="purchase_amount", palette="Set2", ax=ax1)
ax1.set_title("Box Plot")
ax1.set_xlabel("Category")
ax1.set_ylabel("Purchase Amount (£)")
ax1.tick_params(axis="x", rotation=30)

# Violin plot (smoother — shows the full shape of distribution)
sns.violinplot(data=df, x="product_category", y="purchase_amount", palette="Set3", ax=ax2)
ax2.set_title("Violin Plot")
ax2.set_xlabel("Category")
ax2.set_ylabel("Purchase Amount (£)")
ax2.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("charts/day10_05_violin_box.png", dpi=100)
print("Violin/box plot saved → charts/day10_05_violin_box.png")
plt.close()


# ==============================================================
# SECTION 6: SEABORN KDE PLOT — SMOOTH DISTRIBUTION
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: KDE PLOT (DENSITY CURVE)")
print("=" * 60)

# KDE = Kernel Density Estimate — a smooth version of a histogram
# Great for comparing distributions between groups

fig, ax = plt.subplots(figsize=(9, 5))
for gender in df["gender"].unique():
    subset = df[df["gender"] == gender]
    sns.kdeplot(subset["purchase_amount"], label=gender, fill=True, alpha=0.3, ax=ax)

ax.set_title("Purchase Amount Distribution by Gender (KDE)", fontsize=13, fontweight="bold")
ax.set_xlabel("Purchase Amount (£)")
ax.set_ylabel("Density")
ax.legend(title="Gender")

plt.tight_layout()
plt.savefig("charts/day10_06_kde_plot.png", dpi=100)
print("KDE plot saved → charts/day10_06_kde_plot.png")
plt.close()


# ==============================================================
# SECTION 7: SEABORN HEATMAP — CORRELATION MATRIX
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: HEATMAP")
print("=" * 60)

numeric_cols = ["age","income","purchase_amount","discount_pct",
                "customer_rating","num_purchases_last_year","is_member","returns_count"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))   # Only show lower triangle (avoid duplicates)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap (Lower Triangle)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("charts/day10_07_heatmap.png", dpi=100)
print("Heatmap saved → charts/day10_07_heatmap.png")
plt.close()


# ==============================================================
# SECTION 8: FULL DASHBOARD — ALL 8 CHARTS IN ONE FIGURE
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 8: FULL VISUALIZATION DASHBOARD")
print("=" * 60)

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Customer Shopping Data — Full Visualization Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

# Chart 1: Purchase amount histogram
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(df["purchase_amount"], bins=15, color="steelblue", edgecolor="white")
ax1.set_title("Purchase Amount Distribution")
ax1.set_xlabel("Amount (£)")

# Chart 2: Bar chart — avg purchase by category
ax2 = fig.add_subplot(3, 3, 2)
cat_avg = df.groupby("product_category")["purchase_amount"].mean().sort_values()
ax2.barh(cat_avg.index, cat_avg.values, color="coral")
ax2.set_title("Avg Purchase by Category")
ax2.set_xlabel("Amount (£)")

# Chart 3: Income distribution
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(df["income"], bins=15, color="lightgreen", edgecolor="white")
ax3.set_title("Income Distribution")
ax3.set_xlabel("Income (£)")

# Chart 4: Scatter — income vs purchase
ax4 = fig.add_subplot(3, 3, 4)
ax4.scatter(df["income"], df["purchase_amount"], alpha=0.6, color="purple", s=40)
ax4.set_title("Income vs Purchase Amount")
ax4.set_xlabel("Income (£)")
ax4.set_ylabel("Purchase (£)")

# Chart 5: Box plot by gender
ax5 = fig.add_subplot(3, 3, 5)
sns.boxplot(data=df, x="gender", y="purchase_amount", palette="Set2", ax=ax5)
ax5.set_title("Purchase by Gender")

# Chart 6: City counts
ax6 = fig.add_subplot(3, 3, 6)
df["city"].value_counts().plot(kind="bar", ax=ax6, color="teal", edgecolor="black")
ax6.set_title("Customers per City")
ax6.set_xlabel("")
ax6.tick_params(axis="x", rotation=30)

# Chart 7: Age distribution
ax7 = fig.add_subplot(3, 3, 7)
ax7.hist(df["age"], bins=12, color="orange", edgecolor="white")
ax7.set_title("Age Distribution")
ax7.set_xlabel("Age")

# Chart 8: Member vs Non-Member avg spend
ax8 = fig.add_subplot(3, 3, 8)
member_avg = df.groupby("is_member")["purchase_amount"].mean()
ax8.bar(["Non-Member", "Member"], member_avg.values, color=["salmon","steelblue"], edgecolor="black")
ax8.set_title("Member vs Non-Member Spend")
ax8.set_ylabel("Avg Purchase (£)")

# Chart 9: Discount vs Rating scatter
ax9 = fig.add_subplot(3, 3, 9)
ax9.scatter(df["discount_pct"], df["customer_rating"], alpha=0.6, color="darkgreen", s=50)
ax9.set_title("Discount % vs Customer Rating")
ax9.set_xlabel("Discount %")
ax9.set_ylabel("Rating (1-5)")

plt.tight_layout()
plt.savefig("charts/day10_08_full_dashboard.png", dpi=100, bbox_inches="tight")
print("Full dashboard saved → charts/day10_08_full_dashboard.png")
plt.close()


# ==============================================================
# CHOOSING THE RIGHT CHART
# ==============================================================

print("\n" + "=" * 60)
print("WHICH CHART TYPE TO USE WHEN?")
print("=" * 60)

chart_guide = [
    ("Histogram",   "Distribution of ONE numeric column"),
    ("Bar Chart",   "Compare VALUES across categories"),
    ("Scatter Plot","Relationship between TWO numeric columns"),
    ("Line Plot",   "Trends OVER TIME"),
    ("Box Plot",    "Distribution + outliers, compare groups"),
    ("Violin Plot", "Full distribution shape, compare groups"),
    ("Heatmap",     "Correlation between MANY columns at once"),
    ("KDE Plot",    "Smooth distribution, compare groups"),
    ("Pie Chart",   "Parts of a WHOLE (use sparingly!)"),
]

print(f"\n{'Chart Type':<14} {'Best For'}")
print("-" * 65)
for chart, use in chart_guide:
    print(f"  {chart:<14} → {use}")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 10 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Matplotlib: figure → axes → plot() / hist() / scatter() / bar()")
print("✓ Seaborn: boxplot, violinplot, kdeplot, heatmap, pairplot")
print("✓ Always label: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()")
print("✓ Save charts: plt.savefig('file.png', dpi=100)")
print("✓ Choose chart by: what question you're trying to answer")
print("\nTomorrow: Statistical Analysis & Hypothesis Testing!")

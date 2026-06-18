# ==============================================================
# DAY 11 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# Course: ML Fundamentals | Week 3
# ==============================================================

# Statistics lets us ask RIGOROUS questions about data:
#   "Is the difference I see REAL, or just random chance?"
# Hypothesis testing gives us a formal yes/no answer with a probability.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats   # scipy.stats has all the statistical tests we need
import os

print("=" * 60)
print("DAY 11: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
print("=" * 60)

df = pd.read_csv("../data/day06_engineered.csv")
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")


# ==============================================================
# SECTION 1: DESCRIPTIVE STATISTICS REVIEW
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 60)

# Descriptive statistics DESCRIBE and SUMMARISE the data.
# They don't make predictions — they just tell you what IS.

col = "purchase_amount"

mean   = df[col].mean()      # Average — add all values, divide by count
median = df[col].median()    # Middle value when sorted
mode   = df[col].mode()[0]   # Most frequent value
std    = df[col].std()       # Standard deviation — how spread out the data is
variance = df[col].var()     # Variance = std² (spread, but squared)

print(f"\nDescriptive stats for '{col}':")
print(f"  Mean:     £{mean:.2f}")
print(f"  Median:   £{median:.2f}")
print(f"  Mode:     £{mode:.2f}")
print(f"  Std Dev:  £{std:.2f}")
print(f"  Variance: {variance:.2f}")
print(f"  Min:      £{df[col].min():.2f}")
print(f"  Max:      £{df[col].max():.2f}")
print(f"  Range:    £{df[col].max() - df[col].min():.2f}")

# SKEWNESS — is the distribution symmetric?
skew = df[col].skew()
print(f"\n  Skewness: {skew:.3f}")
if skew > 0.5:
    print("  → Right-skewed (tail on the right — most values are low, a few are very high)")
elif skew < -0.5:
    print("  → Left-skewed (tail on the left — most values are high, a few are very low)")
else:
    print("  → Approximately symmetric (looks like a bell curve)")


# ==============================================================
# SECTION 2: PROBABILITY DISTRIBUTIONS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: PROBABILITY DISTRIBUTIONS")
print("=" * 60)

# A DISTRIBUTION describes how values are spread across a range.

# NORMAL DISTRIBUTION (Bell Curve):
#   - Most values cluster around the mean
#   - Symmetric (equal on both sides)
#   - Examples: heights, IQ scores, measurement errors

# Generate a normal distribution for illustration
np.random.seed(42)   # seed=42 makes random numbers reproducible
normal_data = np.random.normal(loc=500, scale=150, size=1000)
# loc = mean, scale = std deviation, size = number of samples

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Normal vs Skewed Distribution", fontsize=13, fontweight="bold")

# Normal distribution
ax1.hist(normal_data, bins=30, color="steelblue", edgecolor="white", density=True)
ax1.set_title("Normal Distribution (Bell Curve)\nmean=500, std=150")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")

# Real purchase_amount (likely skewed)
ax2.hist(df["purchase_amount"], bins=20, color="coral", edgecolor="white", density=True)
ax2.set_title(f"Purchase Amount (Right-Skewed)\nskewness={df['purchase_amount'].skew():.2f}")
ax2.set_xlabel("Purchase Amount (£)")

plt.tight_layout()
plt.savefig("charts/day11_distributions.png", dpi=100)
print("Distribution chart saved → charts/day11_distributions.png")
plt.close()


# ==============================================================
# SECTION 3: HYPOTHESIS TESTING — THE FRAMEWORK
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: HYPOTHESIS TESTING FRAMEWORK")
print("=" * 60)

# HYPOTHESIS TESTING is a formal way to ask:
#   "Is the difference I see in my data REAL, or could it be random chance?"

# THE WORKFLOW:
#   1. H0 (Null Hypothesis)  — "There is NO difference / effect"
#   2. H1 (Alternative Hypothesis) — "There IS a difference / effect"
#   3. Choose significance level α (alpha) — usually 0.05 (5%)
#   4. Run the statistical test → get a P-VALUE
#   5. Decision:
#       p-value < α  → REJECT H0 → The difference is statistically significant!
#       p-value ≥ α  → FAIL TO REJECT H0 → Not enough evidence

# THE P-VALUE:
#   p-value = probability of seeing this result IF H0 is true (just by chance)
#   Small p-value (< 0.05) → very unlikely by chance → something REAL is happening

print("\nHypothesis Testing Steps:")
print("  1. H0: No difference (null hypothesis)")
print("  2. H1: There IS a difference (alternative hypothesis)")
print("  3. Set α = 0.05 (5% significance level)")
print("  4. Run test → get p-value")
print("  5. p < 0.05 → reject H0 (result is statistically significant)")
print("     p ≥ 0.05 → fail to reject H0 (not enough evidence)")


# ==============================================================
# SECTION 4: T-TEST — ARE TWO GROUP MEANS DIFFERENT?
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: T-TEST")
print("=" * 60)

# T-TEST compares the MEAN of two groups.
# Question: "Do male and female customers spend different amounts on average?"

alpha = 0.05   # Our significance threshold

# Separate the data into two groups
male_spend   = df[df["gender"] == "Male"]["purchase_amount"]
female_spend = df[df["gender"] == "Female"]["purchase_amount"]

print(f"\nMale   — n={len(male_spend)},   mean=£{male_spend.mean():.2f},   std=£{male_spend.std():.2f}")
print(f"Female — n={len(female_spend)}, mean=£{female_spend.mean():.2f}, std=£{female_spend.std():.2f}")

# Run the independent two-sample t-test
# H0: mean_male = mean_female (no difference)
# H1: mean_male ≠ mean_female (there IS a difference)
t_stat, p_value = stats.ttest_ind(male_spend, female_spend)

print(f"\nTwo-Sample T-Test: Male vs Female purchase amount")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4f}")
print(f"  α (alpha):   {alpha}")

if p_value < alpha:
    print(f"\n  → p ({p_value:.4f}) < α ({alpha}) → REJECT H0")
    print("  → The difference in spending between genders is STATISTICALLY SIGNIFICANT.")
else:
    print(f"\n  → p ({p_value:.4f}) ≥ α ({alpha}) → FAIL TO REJECT H0")
    print("  → Not enough evidence that genders spend differently.")

# Visualise the comparison
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df[df["gender"].isin(["Male","Female"])],
            x="gender", y="purchase_amount", palette=["steelblue","coral"], ax=ax)
ax.set_title(f"Purchase Amount by Gender\n(t={t_stat:.2f}, p={p_value:.4f})", fontsize=12)
ax.set_xlabel("Gender")
ax.set_ylabel("Purchase Amount (£)")
plt.tight_layout()
plt.savefig("charts/day11_ttest_gender.png", dpi=100)
print("\nT-test chart saved → charts/day11_ttest_gender.png")
plt.close()


# ==============================================================
# SECTION 5: CHI-SQUARE TEST — ARE TWO CATEGORICAL VARIABLES RELATED?
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: CHI-SQUARE TEST")
print("=" * 60)

# CHI-SQUARE TEST checks if two CATEGORICAL columns are INDEPENDENT.
# Question: "Is there a relationship between gender and product category?"

# H0: Gender and product_category are INDEPENDENT (no relationship)
# H1: They ARE related

# First, build a contingency table (cross-tab)
contingency_table = pd.crosstab(df["gender"], df["product_category"])
print("\nContingency Table (Gender × Product Category):")
print(contingency_table)

# Run chi-square test
chi2_stat, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
# chi2_stat = the test statistic
# p_chi     = p-value
# dof       = degrees of freedom
# expected  = what we'd expect if H0 were true

print(f"\nChi-Square Test: Gender vs Product Category")
print(f"  χ² statistic: {chi2_stat:.4f}")
print(f"  p-value:      {p_chi:.4f}")
print(f"  Degrees of freedom: {dof}")

if p_chi < alpha:
    print(f"\n  → p ({p_chi:.4f}) < α ({alpha}) → REJECT H0")
    print("  → Gender and product category ARE related (not independent).")
else:
    print(f"\n  → p ({p_chi:.4f}) ≥ α ({alpha}) → FAIL TO REJECT H0")
    print("  → No significant relationship between gender and product category.")


# ==============================================================
# SECTION 6: ANOVA — COMPARING MORE THAN TWO GROUPS
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: ANOVA TEST (Three or More Groups)")
print("=" * 60)

# ANOVA (Analysis of Variance) extends the t-test to 3+ groups.
# Question: "Do spending amounts differ significantly across all cities?"

# H0: All city means are equal (no difference)
# H1: At least one city's mean is different

# Create a list of purchase amounts, one per city
city_groups = [group["purchase_amount"].values for _, group in df.groupby("city")]
city_names  = df["city"].unique().tolist()

f_stat, p_anova = stats.f_oneway(*city_groups)   # * unpacks the list
# f_stat = F-statistic (ratio of between-group variance to within-group variance)
# p_anova = p-value

print("\nCity groups:")
for city, group in df.groupby("city"):
    print(f"  {city:<12} → n={len(group)}, mean=£{group['purchase_amount'].mean():.0f}")

print(f"\nOne-Way ANOVA: Purchase Amount across Cities")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value:     {p_anova:.4f}")

if p_anova < alpha:
    print(f"\n  → p ({p_anova:.4f}) < α ({alpha}) → REJECT H0")
    print("  → Spending differs significantly across cities.")
else:
    print(f"\n  → p ({p_anova:.4f}) ≥ α ({alpha}) → FAIL TO REJECT H0")
    print("  → No significant difference in spending across cities.")


# ==============================================================
# SECTION 7: ASSIGNMENT EXERCISE — SURVIVAL RATES BY GENDER
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 7: EXERCISE — ARE PURCHASE RATES DIFFERENT BY MEMBER STATUS?")
print("=" * 60)

# In this dataset, we test: do members make more purchases?
# H0: mean num_purchases_last_year is same for members and non-members
# H1: members have MORE purchases

members     = df[df["is_member"] == 1]["num_purchases_last_year"]
non_members = df[df["is_member"] == 0]["num_purchases_last_year"]

print(f"\nMembers:     n={len(members)},     mean={members.mean():.2f}")
print(f"Non-members: n={len(non_members)}, mean={non_members.mean():.2f}")

# One-tailed t-test (alternative='greater' = we expect members to be higher)
t_stat2, p_one_tail = stats.ttest_ind(members, non_members, alternative="greater")

print(f"\nT-Test (one-tailed): Members vs Non-Members — num_purchases_last_year")
print(f"  t-statistic: {t_stat2:.4f}")
print(f"  p-value:     {p_one_tail:.4f}")

if p_one_tail < alpha:
    print(f"\n  → REJECT H0 — Members make significantly MORE purchases than non-members.")
else:
    print(f"\n  → FAIL TO REJECT H0 — Not enough evidence that members buy more often.")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 11 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Descriptive stats: mean, median, mode, std, skewness")
print("✓ Normal distribution = bell curve (symmetric, mean=median)")
print("✓ Hypothesis test: H0 (null) vs H1 (alternative)")
print("✓ p-value < 0.05 → reject H0 → result is statistically significant")
print("✓ T-test: compare MEANS of two groups")
print("✓ Chi-square: test independence of two CATEGORICAL variables")
print("✓ ANOVA: compare means across THREE or more groups")
print("\nTomorrow: Final Graded Assignment — End-to-End Pipeline!")

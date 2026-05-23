# Chapter 5 — Statistics

## Chapter Overview

Statistics refers to the mathematics and techniques with which we understand data. This chapter introduces foundational statistical concepts for describing, summarizing, and exploring datasets.

Rather than diving into formal theory, Joel Grus focuses on practical tools: computing central tendencies, measuring dispersion, and understanding correlation. The emphasis is on building intuition for what these metrics mean, when to use them, and — critically — what they don't tell you.

The chapter uses the DataSciencester user data (friend counts, daily minutes) as a running example to ground abstract concepts in concrete analysis.

The chapter introduces:
- Measures of central tendency (mean, median, mode, quantiles)
- Measures of dispersion (range, variance, standard deviation, IQR)
- Correlation and covariance
- Common pitfalls like Simpson's Paradox
- The difference between correlation and causation

**The key lesson:**

> Statistics helps describe and understand data. It doesn't give you truth — it gives you a structured way to ask better questions.

---

## Learning Objectives

After completing this chapter, I should be able to:

- Compute and interpret measures of central tendency (mean, median, mode, quantiles)
- Calculate and understand measures of dispersion (range, variance, standard deviation, IQR)
- Compute covariance and correlation to assess relationships between variables
- Recognize how outliers affect statistical measures
- Identify pitfalls like Simpson's Paradox and spurious correlation
- Distinguish between correlation and causation
- Apply basic exploratory data analysis (EDA) techniques to one- and two-dimensional data
- Explain why statistics matters in machine learning

---

## Key Terms

| Term | Definition |
|------|------------|
| **Mean** | The average value; sum of values divided by count |
| **Median** | The middle value after sorting; robust to outliers |
| **Mode** | The most frequent value(s) in the data |
| **Quantile** | A value below which a certain percentage of data falls |
| **Percentile** | Quantile expressed as a percentage (e.g., 90th percentile) |
| **Range** | Maximum value minus minimum value |
| **Variance** | Average squared deviation from the mean; measures spread |
| **Standard Deviation** | Square root of variance; in original units |
| **Interquartile Range (IQR)** | Difference between 75th and 25th percentiles; robust measure of spread |
| **Dispersion** | How spread out or concentrated data is |
| **Covariance** | Measures how two variables vary together |
| **Correlation** | Unitless measure of linear relationship strength (from -1 to +1) |
| **Outlier** | An unusual or extreme value that differs significantly from others |
| **De-meaning** | Subtracting the mean from each data point |
| **Simpson's Paradox** | Trend that appears in groups but reverses when groups are combined |
| **Confounding Variable** | A hidden variable that influences both variables being studied |
| **Distribution** | How values are spread across a range |

---

## Mental Model: Statistics as a Lens for Data

> Statistics doesn't give you truth — it gives you a structured way to ask better questions.

Think of statistical measures as different lenses for examining data:

```txt
Raw Data → Summary → Patterns → Understanding → Better Decisions
```

Statistics helps answer questions like:
- **Central tendency** → "What is typical?" (Where is the center of my data?)
- **Dispersion** → "How spread out is the data?" (How much variation exists?)
- **Correlation** → "Are variables related?" (Do two variables move together?)
- **Position** → "Where does a value rank?" (What percentile is it in?)
- **Anomalies** → "Are there unusual values?" (What looks suspicious?)

```txt
Good statistics = Right metric + Context + Awareness of limitations
```

The goal is not to compute numbers mechanically, but to extract insight while avoiding misinterpretation.

---

## Key Concepts

### 1. Describing a Single Set of Data

Start with the basics: count, min, max, sorted order.

```python
num_friends = [100, 49, 41, 40, 25, ...]  # friend counts

num_points = len(num_friends)          # 204
largest_value = max(num_friends)       # 100
smallest_value = min(num_friends)      # 1
sorted_values = sorted(num_friends)
second_largest = sorted_values[-2]     # 49
```

**Why it matters:** These basics reveal data quality issues (e.g., impossible values like negative friend counts) and set the stage for deeper analysis.

---

### 2. Central Tendencies: Finding the Center

These describe **where the center of the data is**.

#### Mean (Average)

The mean is the arithmetic average:

```python
def mean(x):
    return sum(x) / len(x)

mean(num_friends)  # ~7.33
```

- **Sensitive to outliers:** One extreme value can shift the mean significantly
- **Useful when:** Data is roughly symmetric and outliers are rare
- **Problem example:** `[10, 20, 1000]` → Mean = 343.3, which doesn't represent any actual value

#### Median (Middle Value)

The median is the middle value after sorting:

```python
def median(v):
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        return sorted_v[midpoint]
    else:
        lo, hi = midpoint - 1, midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

median(num_friends)  # 6.0
```

- **Robust to outliers:** Changing an extreme value doesn't affect the median
- **Better for:** Skewed distributions (e.g., income data, house prices)
- **For even counts:** Average of the two middle values

#### Mode (Most Frequent Value)

The mode is the most common value:

```python
from collections import Counter

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]

mode(num_friends)  # [1, 6]  (bimodal)
```

- **Useful for:** Categorical data, identifying frequent patterns
- **May have multiple modes:** Bimodal, multimodal distributions

#### Quantiles (Percentiles)

Quantiles generalize the median for any percentage:

```python
def quantile(x, p):
    p_index = int(p * len(x))
    return sorted(x)[p_index]

quantile(num_friends, 0.10)  # 1  (10th percentile)
quantile(num_friends, 0.50)  # 6  (50th percentile = median)
quantile(num_friends, 0.75)  # 9  (75th percentile, Q3)
```

- Median = 50th percentile
- 90th percentile means "higher than 90% of observations"
- Useful for rankings, salary distributions, performance benchmarks

---

### 3. Dispersion: Measuring Spread

These describe **how spread out data is**.

#### Range

```python
def data_range(x):
    return max(x) - min(x)

data_range(num_friends)  # 99
```

- Simple but **highly sensitive to outliers**
- Doesn't reflect how data is distributed between min and max
- Best used as a quick sanity check, not a primary metric

#### Variance

Variance measures how far data points are from the mean:

```python
def de_mean(x):
    """Translate x by subtracting its mean (so result has mean 0)."""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)  # Sample variance

variance(num_friends)  # ~81.54
```

- **Units are squared** (e.g., "friends²"), making direct interpretation difficult
- **Low variance:** Data clustered close to the mean
- **High variance:** Data widely spread out
- **Important in:** Feature scaling, ML optimization, uncertainty analysis

#### Standard Deviation

Standard deviation brings variance back to original units:

```python
import math

def standard_deviation(x):
    return math.sqrt(variance(x))

standard_deviation(num_friends)  # ~9.03
```

- **Same units as original data** — interpretable as "typical distance from mean"
- Most values fall within ±1 standard deviation of the mean (for normal distributions)
- Still **sensitive to outliers**, like variance

#### Interquartile Range (IQR)

```python
def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

interquartile_range(num_friends)  # 6
```

- **Robust to outliers:** Based on the middle 50% of data
- Often used with box plots to identify outliers
- Represents the range where the central half of data lies

---

### 4. Correlation: Measuring Relationships

These describe **whether and how variables move together**.

#### Covariance

```python
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

covariance(num_friends, daily_minutes)  # ~22.43
```

- Measures how two variables vary together
- **Positive:** Variables move in the same direction
- **Negative:** Variables move in opposite directions
- **Near zero:** Little or no linear relationship
- **Hard to interpret:** Units are product of both variables' units

#### Correlation Coefficient (Pearson's r)

```python
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0  # If no variation, correlation is 0

correlation(num_friends, daily_minutes)  # ~0.25 (weak positive)
```

- **Unitless:** Always between -1 and +1
- **+1:** Perfect positive linear relationship
- **-1:** Perfect negative linear relationship
- **0:** No linear relationship
- **Critical:** Extremely sensitive to outliers

```python
# Example: removing one outlier can dramatically change correlation
outlier = num_friends.index(100)
num_friends_good = [x for i, x in enumerate(num_friends) if i != outlier]
daily_minutes_good = [x for i, x in enumerate(daily_minutes) if i != outlier]

correlation(num_friends_good, daily_minutes_good)  # ~0.57 (jumps from 0.25!)
```

---

### 5. Pitfalls & Caveats

#### Simpson's Paradox

**A correlation can reverse when you account for a confounding variable.**

Example from the book: West Coast data scientists appear friendlier (avg 8.2 friends) than East Coast (6.5). But when controlling for PhD status:
- PhD holders: East Coast (3.2) > West Coast (3.1)
- Non-PhD: East Coast (13.4) > West Coast (10.9)

The paradox occurs because West Coast has more PhDs (who have fewer friends on average), pulling down the overall average.

**Lesson:** Always check for confounding variables before drawing conclusions from aggregated data.

#### Correlation ≠ Causation

A strong correlation between `num_friends` and `daily_minutes` could mean:
- More friends → More time on site (direct causation)
- More time on site → More friends (reverse causation)
- A third factor (passion for data science) causes both (confounding)
- Pure coincidence (spurious correlation)

**Lesson:** Correlation suggests hypotheses; controlled experiments (A/B tests) test causation.

#### Zero Correlation ≠ No Relationship

Variables can have a strong non-linear relationship with zero linear correlation.

Example: `x = [-2, -1, 0, 1, 2]`, `y = [4, 1, 0, 1, 4]` (y = x²) → correlation = 0.

**Lesson:** Always visualize your data; correlation only captures linear relationships.

#### Outliers Can Distort Everything

One extreme value can dramatically change:
- Mean (pulls it toward the outlier)
- Variance and standard deviation (inflates them)
- Correlation (can create or hide relationships)

**Lesson:** Investigate outliers before computing summary statistics.

---

## Choosing the Right Metric

### Central Tendency

| Situation | Preferred Metric | Why |
|-----------|-----------------|-----|
| Symmetric, no outliers | Mean | Most efficient; uses all data |
| Skewed or outliers present | Median | Robust to extreme values |
| Categorical/discrete data | Mode | Identifies most common category |
| Understanding distribution shape | Quantiles | Shows full range of positions |

### Dispersion

| Situation | Preferred Metric | Why |
|-----------|-----------------|-----|
| Quick first look | Range | Simplest to understand |
| Need interpretable spread | Standard Deviation | Same units as data |
| Outliers present | IQR | Robust to extreme values |
| Analyzing distribution | All three together | Each reveals different aspects |

---

## Important Takeaways

### Statistics Summarises Data
Instead of analyzing thousands of individual numbers, statistics provides concise summaries that reveal patterns, typical values, and unusual observations.

### Mean Can Be Misleading
Especially with outliers or skewed distributions. The median is often more representative of "typical" values.

### Variability Matters as Much as Averages
Two datasets can have the same mean but very different spreads. Understanding dispersion is essential for:
- Assessing risk and uncertainty
- Detecting anomalies
- Evaluating model consistency

### Correlation Helps Find Relationships
Useful in feature engineering, exploratory analysis, and model building. But always remember:
- Visualize before computing correlation
- Check for outliers that may be driving the relationship
- Watch for Simpson's Paradox and confounding variables
- Correlation does not imply causation

### Statistics Supports ML at Every Stage
From data cleaning to feature engineering to model evaluation to production monitoring — statistical thinking is woven throughout the ML lifecycle.

---

## Connections to AI/ML Engineering

This chapter is foundational for virtually every stage of the ML workflow:

| Statistical Concept | ML/Data Science Application |
|--------------------|---------------------------|
| Mean/median imputation | Handling missing values in preprocessing pipelines |
| Standard deviation | Feature scaling (standardization/z-score normalization) |
| IQR | Robust outlier detection in data cleaning |
| Correlation | Feature selection, detecting multicollinearity in regression models |
| Quantiles | Understanding prediction distributions, calibration |
| Simpson's Paradox | Avoiding biased models by controlling for confounders |
| EDA techniques | Understanding data before model building, debugging model failures |
| De-meaning | Centering data for PCA, gradient descent optimization |

### Specific Applications:

**Exploratory Data Analysis (EDA):**
- Understand distributions, outliers, and spread before modeling
- Identify data quality issues early

**Feature Engineering:**
- Identify which features need scaling
- Detect anomalies and decide on transformations
- Assess feature importance through correlation with target

**Model Evaluation:**
- Compare model performance across subgroups (watch for Simpson's Paradox)
- Analyze residual distributions
- Use statistical tests to compare models

**Production Monitoring:**
- Track mean drift and variance drift
- Monitor distribution changes over time
- Detect data quality degradation

Strong statistical intuition enables:
- Better feature engineering and selection
- More robust data cleaning and preprocessing
- Clearer interpretation of model coefficients and results
- Avoiding common pitfalls like overfitting to spurious correlations

---

## Questions / Confusions

Things I still want to understand better:

- [ ] When should I use sample variance (n-1) vs population variance (n)?
- [ ] How do I handle correlation with non-linear relationships (e.g., Spearman's rank correlation)?
- [ ] What are robust alternatives to mean and standard deviation for heavily skewed data?
- [ ] How do I formally test for Simpson's Paradox in my datasets?
- [ ] What's the relationship between correlation and causation in modern causal inference frameworks?
- [ ] How does correlation differ from mutual information?
- [ ] What statistical knowledge matters most for ML engineering interviews?

---

## Personal Reflections

### What surprised me?
How dramatically a single outlier can change correlation — from 0.25 to 0.57 just by removing one test account. This reinforces the importance of data cleaning and outlier investigation before drawing any conclusions. Also surprising: how misleading averages can be when data is skewed.

### What stood out?
Simpson's Paradox is a powerful reminder that aggregated data can hide — or even reverse — true relationships. As an ML engineer, I'll need to be vigilant about stratifying data and controlling for confounders when analyzing model performance across subgroups. Also: two datasets can have identical means but completely different behavior.

### How does this connect to my roadmap?
Statistics is the foundation of model evaluation, feature engineering, and data validation. Whether I'm:
- Diagnosing why a model performs poorly on a subgroup (Simpson's Paradox)
- Scaling features for a neural network (standard deviation)
- Detecting anomalies in production data (IQR, z-scores)
- Selecting features to avoid multicollinearity (correlation)
- Designing A/B tests to evaluate model changes
- Monitoring production ML systems for drift

…the concepts from this chapter will be running under the hood. Mastering them now means I'll spend less time debugging statistical misunderstandings and more time building robust systems.

---

## Chapter Summary (In My Own Words)

Chapter 5 introduces practical statistical tools for describing and exploring data. Central tendencies (mean, median, mode, quantiles) tell us where data is centered; dispersion metrics (range, variance, standard deviation, IQR) tell us how spread out it is; correlation measures linear relationships between variables.

The key lessons are:

1. **No single metric tells the whole story** — use multiple lenses (mean AND median, variance AND IQR)
2. **Outliers can distort statistics** — always visualize and investigate before computing summaries
3. **Correlation suggests but doesn't prove causation** — control for confounders, watch for Simpson's Paradox
4. **Statistics is a tool for asking better questions**, not for delivering final answers
5. **Visualization + statistics** is more powerful than either alone

These fundamentals form the bedrock of exploratory data analysis, feature engineering, and model evaluation in machine learning. As I build more complex systems, I'll return to these concepts again and again.

> "The plural of anecdote is not data, but the plural of data is not insight — without statistics."

---

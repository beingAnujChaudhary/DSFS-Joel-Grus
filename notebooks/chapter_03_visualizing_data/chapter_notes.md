# Chapter 3 — Visualizing Data

## Chapter Overview

This chapter introduces data visualization as a fundamental tool for exploring and communicating data. Joel Grus demonstrates that while creating visualizations is technically easy, producing *good* ones requires intentionality and clear thinking.

The chapter primarily focuses on `matplotlib.pyplot` and covers three core chart types:
- **Bar charts**: For comparing quantities across discrete categories or bucketed numeric values
- **Line charts**: For showing trends over time or ordered sequences
- **Scatterplots**: For visualizing relationships between two paired variables

The emphasis is on clarity, avoiding misleading representations, and using visualization to *understand* data—not just decorate it.

The key lesson:

> Always visualize data before modelling it. A chart is not just a picture—it's an argument.

---

## Learning Objectives

After completing this chapter, I should be able to:

- Understand why data visualization matters before analysis or modelling
- Create basic visualizations using `matplotlib.pyplot`
- Choose the appropriate chart type for a given analytical question
- Customize charts with labels, titles, legends, and axis formatting
- Recognize and avoid common visualization pitfalls (e.g., misleading axes)
- Use visualization to explore data distributions and relationships
- Apply best practices for clear, honest, publication-quality charts

---

## Key Terms

| Term | Definition |
|------|------------|
| Data Visualization | The graphical representation of data to reveal patterns |
| Exploratory Data Analysis (EDA) | Using visuals and statistics to understand data before modelling |
| `matplotlib.pyplot` | Python's foundational plotting library |
| Bar Chart | Compares quantities across discrete categories |
| Histogram | Shows distribution by bucketing numeric data into bins |
| Line Chart | Shows trends over ordered sequences or time |
| Scatter Plot | Visualizes relationships between two paired variables |
| Trend | A general direction in which data moves over time |
| Correlation | A statistical relationship between two variables |
| Outlier | A data point that differs significantly from others |
| Distribution | How values are spread across a range |
| Bucketing/Binning | Grouping continuous data into discrete intervals |
| Axis Labels, Legend, Title | Essential chart elements for clarity |
| Misleading Y-axis | Truncating axes to exaggerate differences |

---

## Mental Model: Visualization as Communication

Think of data visualization as a pipeline from raw data to insight:

```txt
Raw Data
    ↓
Visualization
    ↓
Pattern Recognition
    ↓
Insights
    ↓
Better Decisions
```

Before building models:

> First understand what the data is telling you.

Every visualization should answer a question:
- **What am I trying to show?** (comparison, distribution, relationship, trend)
- **Who is my audience?** (technical team, stakeholders, public)
- **What could be misunderstood?** (axis scaling, omitted context, visual clutter)

**Golden rule:**

```txt
Good visualization = Clear question + Appropriate chart + Honest representation
```

Visualization often reveals hidden issues that statistics alone miss.

---

## Why Visualization Matters

Looking only at numbers can be dangerously misleading.

**Example:** A dataset might show `Average = 50`, but visualization may reveal:
- Skewed data (not normally distributed)
- Multiple clusters (subgroups with different behaviour)
- Outliers (extreme values pulling the average)
- Unusual distributions (bimodal, long-tailed)

Graphs help humans recognize patterns faster than raw tables. Before training any ML model:

```txt
Step 1: Understand the data visually
Step 2: Confirm with statistics
Step 3: Then consider modelling
```

---

## Key Concepts

### 1. matplotlib.pyplot Basics

`matplotlib.pyplot` maintains an internal state to build visualizations step-by-step:

```python
import matplotlib.pyplot as plt

# Basic workflow
plt.plot(x_values, y_values)      # Create chart
plt.title("Chart Title")          # Add title
plt.xlabel("X-axis label")        # Label axes
plt.ylabel("Y-axis label")
plt.show()                        # Display
# plt.savefig("chart.png")        # Or save to file
```

**Key insight**: Start simple, then layer customizations. Each `plt.*` call modifies the current figure.

---

### 2. Bar Charts

Use bar charts to show how a quantity varies across **discrete categories**.

```python
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# Center bars by offsetting x-coordinates
xs = [i + 0.1 for i, _ in enumerate(movies)]
plt.bar(xs, num_oscars)
plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)
plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")
plt.show()
```

#### Bar Charts as Histograms (for Distributions)

Bucket numeric data to show distributions:

```python
from collections import Counter

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)

plt.bar([x - 4 for x in histogram.keys()], histogram.values(), 8)
plt.axis([-5, 105, 0, 5])
plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
```

#### ⚠️ Critical Warning: Never Truncate Y-Axis on Bar Charts

Truncating the y-axis misleads viewers about magnitude differences:

```python
# ❌ Misleading: y-axis starts at 499
plt.axis([2012.5, 2014.5, 499, 506])

# ✅ Honest: y-axis starts at 0
plt.axis([2012.5, 2014.5, 0, 550])
```

Bar charts **must** start at zero because we judge bar lengths by their total extent.

---

### 3. Line Charts

Use line charts to show **trends over ordered data** (time, sequence, model complexity, etc.).

```python
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = range(len(variance))

# Multiple series on one chart with different styles
plt.plot(xs, variance, 'g-', label='variance')      # green solid
plt.plot(xs, bias_squared, 'r-.', label='bias²')    # red dot-dash
plt.plot(xs, total_error, 'b:', label='total error') # blue dotted

plt.legend(loc=9)  # top center
plt.xlabel("model complexity")
plt.title("The Bias-Variance Tradeoff")
plt.show()
```

**Style reference:**
- Line styles: `'-'` solid, `'--'` dashed, `'-.'` dot-dash, `':'` dotted
- Colors: `'r'` red, `'g'` green, `'b'` blue, `'k'` black, `'c'` cyan, `'m'` magenta, `'y'` yellow

**Best for:** Time series data, learning curves, trend analysis

---

### 4. Scatter Plots

Use scatter plots to visualize **relationships between two paired variables**.

```python
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# Annotate each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count, minute_count),
                 xytext=(5, -5),
                 textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()
```

#### ⚠️ Use Comparable Axes When Appropriate

When plotting comparable variables (e.g., two test scores), use `plt.axis("equal")` to avoid misleading scale differences:

```python
# Without equal axes: variation appears misleadingly balanced
plt.scatter(test_1_grades, test_2_grades)

# With equal axes: true variation is visible
plt.scatter(test_1_grades, test_2_grades)
plt.axis("equal")
```

**Best for:** Correlation analysis, cluster detection, outlier identification

---

### 5. Customization Essentials

| Function | Purpose | Example |
|----------|---------|---------|
| `plt.title()` | Add chart title | `plt.title("Sales Trend")` |
| `plt.xlabel()` / `plt.ylabel()` | Label axes | `plt.xlabel("Month")` |
| `plt.legend()` | Add legend for multiple series | `plt.legend(loc='upper left')` |
| `plt.xticks()` / `plt.yticks()` | Customize tick labels | `plt.xticks([0,1,2], ['A','B','C'])` |
| `plt.axis()` | Set axis limits | `plt.axis([0, 10, 0, 100])` |
| `plt.savefig()` | Export to file | `plt.savefig("chart.png", dpi=300)` |
| `plt.annotate()` | Add text annotations | `plt.annotate('peak', xy=(5, 10))` |

**Pro tip**: Use `dpi=300` for publication-quality exports and `bbox_inches='tight'` to trim whitespace.

---

## Chart Selection Guide

| Question Type | Best Chart | Example Use Case |
|--------------|------------|------------------|
| Compare categories | Bar chart | Sales by region, population by country |
| Show distribution | Histogram | Exam scores, income distribution, age spread |
| Show trend over time | Line chart | Stock prices, revenue growth, temperature trends |
| Show relationship between two variables | Scatter plot | Hours studied vs marks, height vs weight |
| Part-to-whole composition | (Not covered in this chapter) | Market share, budget allocation |

**Choosing chart based on question:**

```txt
Trend → Line chart
Comparison → Bar chart
Distribution → Histogram
Relationship → Scatter plot
```

---

## Important Takeaways

### Always Visualize Before Modelling
Never immediately train a model. First, understand the data. Visualizations often expose:
- Missing values
- Outliers
- Strange behaviour
- Feature relationships
- Distribution problems

### Honesty > Aesthetics
- Always start y-axis at 0 for bar charts
- Use `axis("equal")` when comparing similar scales
- Label axes clearly—never assume viewers know units
- Avoid "chart junk" (excessive decoration that doesn't add information)

### Different Charts Solve Different Problems
No single chart works for everything. Choose based on the question being asked and the story you want to tell.

### Correlation ≠ Causation
A scatter plot may show a relationship between two variables. But two variables moving together does not prove one causes the other.

### Exploration vs. Communication
- **Exploratory charts**: Quick, messy, for your eyes only
- **Communication charts**: Polished, labeled, audience-aware
- Don't waste time polishing exploratory visuals

---

## Common Visualization Mistakes

| Mistake | Why It's Harmful | How to Avoid |
|---------|------------------|--------------|
| **Misleading axes** | Truncating y-axis exaggerates differences | Start bar chart y-axis at 0; use `axis("equal")` for comparable scales |
| **Too much information** | Overcrowded graphs become unreadable | Keep it simple; one message per chart |
| **Wrong chart selection** | Wrong chart type obscures the insight | Match chart type to analytical question |
| **Missing labels** | Viewers can't interpret the data | Always label axes, add titles, include legends |
| **Unnecessary decoration** | "Chart junk" distracts from the data | Remove anything that doesn't add information |

---

## Connections to AI/ML Engineering

This chapter connects strongly to my long-term AI/ML engineering goals:

| Visualization Skill | ML Application |
|-------------------|----------------|
| Histograms | Understanding feature distributions, detecting outliers |
| Scatterplots | Exploring feature relationships, correlation analysis, cluster detection |
| Line charts | Monitoring training/validation loss curves, diagnosing overfitting |
| Axis customization | Creating clear dashboards for stakeholders |
| Avoiding misleading charts | Ethical reporting of model results, A/B test outcomes |
| Annotations | Highlighting key insights in model performance reports |

Strong visualization skills enable:
- **Faster debugging** of data pipelines and model behaviour
- **Clearer communication** of model insights to non-technical stakeholders
- **Better intuition** for feature engineering and model selection
- **More compelling presentations** of project results
- **Ethical data representation** that avoids misleading conclusions

---

## Questions / Confusions

Things I still want to understand better:

- [ ] When should I use seaborn instead of matplotlib?
- [ ] When should I use `plt.subplots()` vs multiple `plt.figure()` calls?
- [ ] How do I create interactive visualizations (e.g., with Plotly or Bokeh)?
- [ ] What are best practices for colorblind-accessible palettes?
- [ ] How do I efficiently visualize high-dimensional data (beyond 2D scatterplots)?
- [ ] When is it appropriate to use logarithmic scales?
- [ ] What makes a graph truly misleading vs acceptably simplified?
- [ ] How much visualization is used in production ML systems?
- [ ] When should I visualize vs calculate statistics?

---

## Personal Reflections

### What surprised me?
How much impact small formatting choices (axis limits, tick labels, legend placement) have on how a chart is interpreted. A "technically correct" chart can still mislead if not designed thoughtfully. Visualization can reveal problems that raw numbers completely hide.

### What stood out?
The bias-variance tradeoff visualization elegantly communicates a complex ML concept in one glance. Different chart types answer fundamentally different questions—there's no one-size-fits-all approach. This is the power of good visualization: making abstract ideas concrete and immediately understandable.

### How does this connect to my roadmap?
As I build ML models, I'll need to:
- Visualize training/validation loss curves to diagnose overfitting
- Plot feature distributions to guide preprocessing decisions
- Create clear result summaries for stakeholders who may not understand code
- Avoid misleading visualizations that could lead to poor business decisions
- Use EDA techniques to understand data before modelling

Learning matplotlib now builds a foundation for more advanced tools (seaborn, Plotly, D3.js) later.

---

## Chapter Summary (In My Own Words)

Chapter 3 teaches that visualization is both an art and a science. Using `matplotlib.pyplot`, we can create bar charts (for categories and distributions), line charts (for trends over ordered sequences), and scatterplots (for relationships between paired variables).

The core lessons are:

1. **Always visualize before modelling** — graphs reveal patterns, outliers, and problems that summary statistics miss
2. **Choose the right chart for the question** — bar for comparison, line for trends, scatter for relationships, histogram for distributions
3. **Honesty over aesthetics** — never truncate bar chart y-axes, use equal axes when comparing scales, and label everything clearly
4. **Customize thoughtfully** — every element (title, labels, legend, annotations) should guide understanding, not distract from it

> "A picture is worth a thousand words—but only if it's the right picture, told the right way."

Good visualizations don't just show data—they tell a clear, truthful story that drives insight and better decisions. This skill is foundational for all future work in data science and machine learning.

---

Nice — here’s a compact, practical step-by-step EDA (exploratory data analysis) workflow you can run with **pandas**, **matplotlib** and **seaborn**, plus a reusable script/template and best practices. Copy–paste and adapt to your DataFrame (`df`).

# EDA step-by-step (with code examples)

```python
# common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")          # seaborn style; matplotlib still works underneath
plt.rcParams["figure.dpi"] = 120   # sharper figures
```

### 0) Quick checks (first 30 seconds)

```python
df.shape            # rows, cols
df.head()           # first rows
df.sample(5)        # random rows
df.columns.tolist()
df.info()           # dtypes + non-null counts
df.describe(include='all').T  # summary stats (object + numeric)
```

### 1) Missing values & duplicates

```python
# table of missing %
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
pd.concat([missing, missing_pct.rename("missing_pct")], axis=1).sort_values("missing_pct", ascending=False)

# duplicates
df.duplicated().sum()
```

Visualize missingness:

```python
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing value map")
plt.show()
```

(For big datasets prefer `missingno` library — optional.)

### 2) Data types & conversions

```python
df.select_dtypes(include=['object']).nunique()   # categorical cardinality
# convert date-like, categorical, numeric strings
df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')
df['cat_col'] = df['cat_col'].astype('category')
```

### 3) Univariate analysis — numeric

```python
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])    # distribution
    sns.boxplot(x=df[col].dropna(), ax=axes[1])             # outliers
    axes[0].set_title(f"{col} — distribution")
    axes[1].set_title(f"{col} — boxplot")
    plt.tight_layout(); plt.show()
```

### 4) Univariate analysis — categorical

```python
cat_cols = df.select_dtypes(include=['category','object']).columns.tolist()
for col in cat_cols:
    vc = df[col].value_counts().head(15)   # show top categories
    plt.figure(figsize=(6,3))
    sns.barplot(x=vc.values, y=vc.index)
    plt.title(f"{col} — top values")
    plt.show()
```

### 5) Bivariate (numeric vs numeric)

```python
# scatter + regression
x, y = 'col_x', 'col_y'
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
sns.regplot(data=df, x=x, y=y, scatter=False, ci=None, line_kws={'color':'red'})
plt.title(f"{x} vs {y}")
plt.show()

# hexbin for heavy overplotting
plt.figure(figsize=(6,4))
plt.hexbin(df[x], df[y], gridsize=40)
plt.colorbar(label='count')
plt.xlabel(x); plt.ylabel(y); plt.show()
```

### 6) Categorical vs numeric

```python
sns.boxplot(data=df, x='category_col', y='numeric_col')
plt.xticks(rotation=45)
plt.show()

# groupby summary
df.groupby('category_col')['numeric_col'].agg(['count','mean','median','std']).sort_values('count', ascending=False)
```

### 7) Correlation matrix + heatmap

```python
num = df.select_dtypes(include=[np.number])
corr = num.corr(method='pearson')   # or 'spearman' for monotonic relationships
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8,6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation (pearson)")
plt.show()
```

### 8) Pairwise relationships (small sample)

```python
sample = df[num_cols].sample(min(len(df), 500))
sns.pairplot(sample)    # heavy for many cols — limit columns
plt.show()
```

### 9) Time series (if you have a datetime)

```python
df['date'] = pd.to_datetime(df['date'])
ts = df.set_index('date').resample('M')['target'].mean()
ts.plot(figsize=(10,3)); plt.title("Monthly mean target"); plt.show()

# rolling
ts.rolling(window=3).mean().plot(); plt.title("3-month rolling mean"); plt.show()
```

### 10) Outliers detection (IQR)

```python
def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return series[(series < lower) | (series > upper)]

out = iqr_outliers(df['numeric_col'].dropna())
len(out), out.head()
```

### 11) Feature engineering & aggregation

```python
# dummies
df = pd.get_dummies(df, columns=['small_cat'], drop_first=True)

# pivot / cross-tab
pd.pivot_table(df, index='region', columns='month', values='sales', aggfunc='sum', fill_value=0)
```

### 12) Prepare for modeling

* Check target distribution (imbalance).
* Train/test split with `stratify=` for classification.
* Scale numeric features if using distance-based models.
* Save the EDA plots & artifacts.

# Reusable EDA script (compact)

```python
def quick_eda(df, target=None, sample_n=1000, outdir='eda_output'):
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    sns.set(style='whitegrid')
    os.makedirs(outdir, exist_ok=True)
    # summary
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing': (df.isnull().sum()/len(df)).sort_values(ascending=False).to_dict(),
        'duplicates': int(df.duplicated().sum())
    }
    # save a CSV summary
    pd.DataFrame.from_dict(summary, orient='index').to_csv(os.path.join(outdir,'summary.csv'))
    # numeric quick plots
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num:
        plt.figure(figsize=(6,3))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
        plt.tight_layout(); plt.savefig(os.path.join(outdir,f"hist_{col}.png")); plt.close()
    # correlation
    if len(num) >= 2:
        corr = df[num].corr()
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.tight_layout(); plt.savefig(os.path.join(outdir,'corr.png')); plt.close()
    return summary
```

Run: `summary = quick_eda(df, target='target_col')`

# Best practices & tips (short)

* **Start wide, then deep.** Use quick summaries first, then deep dives on interesting features/relationships.
* **Reproducibility:** set `random_state` for sampling, document transformations, version datasets.
* **Plot readability:** always label axes, add titles, rotate x-ticks for long labels, use log scale for skewed distributions (`plt.xscale('log')`).
* **Large data:** sample (`df.sample(n=10000, random_state=1)`), or use binned/aggregated plots (hexbin, 2D hist).
* **Avoid overplotting:** use `alpha` or `hexbin`/`kde`.
* **Categorical cardinality:** convert high-cardinality strings to `category` dtype; combine rare categories into "Other".
* **Check for data leakage** before modeling (features derived from the target).
* **Document assumptions & decisions** (why you imputed, why you dropped rows, transforms).
* **Automate heavy EDA** with tools (ydata-profiing / pandas_profiling, Sweetviz) when you need a quick first pass — but review results manually.

# Quick cheat-sheet of useful commands

* `df.info(), df.describe(), df.value_counts(), df.isnull().sum()`
* `sns.histplot(), sns.boxplot(), sns.violinplot(), sns.scatterplot(), sns.heatmap(), sns.pairplot()`
* `df.groupby(col)[num].agg(['count','mean','median'])`
* `pd.get_dummies(), pd.to_datetime(), df.resample(), df.rolling()`
* `plt.savefig('file.png', dpi=150)` to persist figures

---

If you want, paste `df.head()` and `df.dtypes` (or list your column names and the target) and I’ll give a **tailored EDA script** showing the exact plots and checks you should run for your dataset.
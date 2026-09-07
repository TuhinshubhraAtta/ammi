# AMMI: Adaptive Micro-Manifold Imputer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

**AMMI (Adaptive Micro-Manifold Imputer)** is a fast, linear-time $\mathcal{O}(n)$ non-parametric tabular imputation framework designed for numerical data. It combines orthonormal random hyperplane projection slicing, empirical Bayes cell-level shrinkage, and correlation-weighted residual projection to provide a middle ground between fast univariate baselines and computationally expensive iterative multivariate imputers:

* **Linear Computational Complexity**: Operates in $\mathcal{O}(L \cdot d \cdot n)$ time without computing pairwise distance matrices ($\mathcal{O}(n^2)$) or running cyclic iterative regressions.
* **Neighborhood Awareness**: Quantizes continuous spaces into local micro-neighborhoods using random projections, capturing non-linear feature clustering.
* **Covariance Preservation**: Restores multivariate correlations in a single pass using regularized empirical correlation tangents, avoiding the correlation collapse common to mean and median imputation.
* **Scikit-Learn Native**: Fully implements the `BaseEstimator` and `TransformerMixin` API for seamless integration into pipelines and grid searches.

---

## Empirical Benchmarks

We evaluated AMMI against standard baseline and iterative imputation methods on synthetic tabular data (50,000 samples $\times$ 4 features with non-linear feature interaction $x_2 = 0.05 x_0^2 + 0.5 x_1$ and 15% missing completely at random).

### Benchmark Results (50,000 Rows $\times$ 4 Features)

| Algorithm | Complexity | Runtime (ms) | Imputation Error (RMSE) | Correlation Distortion (Lower=Better) | Scalable to $n > 10^5$? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global Mean** | $\mathcal{O}(n)$ | 7.6 ms | 2.0380 | 0.0822 | Yes |
| **Global Median** | $\mathcal{O}(n)$ | 11.3 ms | 2.0391 | 0.0823 | Yes |
| **KNN Imputer ($k=5$)** | $\mathcal{O}(n^2)$ | *Crashed (OOM)* | N/A | N/A | No |
| **MICE (BayesianRidge, 5 iter)** | $\mathcal{O}(M \cdot \text{iter} \cdot n)$ | 111.3 ms | 0.6616 | 0.0055 | Moderate |
| **AMMI** | $\mathcal{O}(n)$ | 629.3 ms | 1.2960 | 0.0276 | Yes |

*Benchmarks can be reproduced locally by running `python benchmarks/run_benchmark.py --n-samples 50000`.*

### Summary of Observations

1. **Covariance Preservation**: Univariate methods (mean/median) flatten feature covariances (correlation distortion $\approx 0.082$). AMMI reduces correlation distortion by ~66% down to $0.0276$, preserving multi-feature joint distributions for downstream ML models.
2. **Scalability vs. KNN**: Distance-based methods like standard $k$-nearest neighbors require storing or computing all-pairs distances ($\sim 2.5 \times 10^9$ pairwise calculations at $n = 50,000$), quickly causing memory exhaustion. AMMI partitions samples in linear time.
3. **Single-Pass Simplicity**: Unlike MICE, AMMI does not require cycling round-robin regression models across all features.

---

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/TuhinshubhraAtta/ammi.git
```

Or install locally in editable mode:
```bash
git clone https://github.com/TuhinshubhraAtta/ammi.git
cd ammi
pip install -e .
```

---

## Quickstart

AMMI works with **Pandas DataFrames**, **NumPy ndarrays**, and raw **Python Lists**.

### 1. With Pandas DataFrames
```python
import pandas as pd
import numpy as np
from ammi import AMMI

df = pd.DataFrame({
    'Feature_A': [10.2, 11.5, np.nan, 25.1, 28.4, np.nan],
    'Feature_B': [1.2, 1.8, 1.5, 12.0, 14.5, 13.2],
    'Target_Val': [102.0, 118.0, np.nan, 340.0, 390.0, 375.0]
})

imputer = AMMI(n_projections=4, n_bins=4, shrinkage_tau=3.0, random_state=42)
df_imputed = imputer.fit_transform(df)

# Returns a pandas DataFrame preserving original index, column names, and dtypes
print(df_imputed)
```

### 2. In Scikit-Learn Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from ammi import AMMI

pipeline = Pipeline([
    ('imputer', AMMI(n_projections=4, n_bins=4, random_state=42)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 3. With NumPy Arrays
```python
import numpy as np
from ammi import AMMI

X = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 6.0],
    [3.0, 4.0, 9.0],
    [4.0, 5.0, 12.0]
])

imputer = AMMI(n_projections=2, n_bins=2, random_state=42)
X_clean = imputer.fit_transform(X)
print(X_clean)
```

---

## Methodology

AMMI operates in three statistical stages:

### 1. Orthonormal Random Projection Slicing
To group samples into continuous neighborhoods without pairwise distance comparisons, AMMI generates an orthonormal basis $Q \in \mathbb{R}^{d \times L}$ via QR decomposition on random Gaussian vectors:
$$G \sim \mathcal{N}(0, I), \quad G = Q R, \quad Q^T Q = I_L$$
Normalized observations $Z \in \mathbb{R}^{n \times d}$ are projected onto this subspace:
$$P = Z Q \in \mathbb{R}^{n \times L}$$
For each projection axis $l \in [1, L]$, empirical quantiles partition the continuous projection into $B$ discrete intervals. A sample's cell coordinates are given by:
$$c_i = \left( \text{bin}(P_{i, 1}), \ldots, \text{bin}(P_{i, L}) \right)$$

### 2. Hierarchical Empirical Bayes Cell Shrinkage
Within each micro-cell $c$, sample means $\mu_{c, j}$ are computed for each feature $j$. Because cells can become sparse, cell expectations are regularized toward the marginal median $m_j$ using an empirical Bayes shrinkage weight:
$$w_{c, j} = \frac{N_{c, j}}{N_{c, j} + \tau}$$
where $N_{c, j}$ is the number of observed samples in cell $c$ for feature $j$, and $\tau \ge 0$ is a pseudo-count regularization parameter.

### 3. Correlation-Weighted Residual Projection Prior
To restore joint multivariate dependencies without cyclic regressions, AMMI calculates the empirical correlation matrix $R \in \mathbb{R}^{d \times d}$ across observed feature pairs. For an instance $i$ with missing feature $j$ and observed feature set $\mathcal{O}_i$, the residual projection is:
$$\Delta x_{ij} = \sigma_j \cdot \frac{\sum_{k \in \mathcal{O}_i} R_{j, k} Z_{ik}}{\sum_{k \in \mathcal{O}_i} |R_{j, k}| + \epsilon}$$
The final imputed value adaptively blends the local manifold expectation with the global regularized covariance prior:
$$\hat{X}_{ij} = w_{c, j} \mu_{c, j} + (1 - w_{c, j}) (m_j + \Delta x_{ij})$$
When a cell is densely populated ($w \to 1$), local non-linear geometry dominates. In sparse regions ($w \to 0$), the estimator smoothly transitions to the global correlation prior.

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `n_projections` | `int` | `4` | Number of orthonormal random projection hyperplanes. Higher values partition data into finer neighborhoods, but exponentially increase cell sparsity ($n_{\text{bins}}^{n_{\text{projections}}}$). |
| `n_bins` | `int` | `4` | Number of quantile subdivisions per projection axis. |
| `shrinkage_tau` | `float` | `3.0` | Regularization parameter for empirical Bayes cell shrinkage. Higher values shrink sparse cells more heavily toward the global prior. |
| `random_state` | `int, None` | `42` | Seed for reproducible random projection generation. |

---

## Limitations & Considerations

1. **Dimensionality Scaling**: The total number of theoretical micro-cells grows as $\mathcal{O}(n_{\text{bins}}^{n_{\text{projections}}})$. For higher-dimensional datasets ($d > 50$), keep $n_{\text{projections}} \le 6$ to prevent excessive cell sparsity.
2. **Feature Types**: AMMI is currently designed for continuous numerical features. Categorical variables should be pre-encoded or handled via target encoding prior to imputation.
3. **Missing Completely at Random (MCAR) / Missing at Random (MAR)**: AMMI assumes missingness is conditionally independent of unobserved values given observed features. Non-ignorable missingness (MNAR) may require domain-specific correction.

---

## License

This project is licensed under the [MIT License](LICENSE).

# AMMI: Adaptive Micro-Manifold Imputer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**AMMI (Adaptive Micro-Manifold Imputer)** is a fast, linear-time $\mathcal{O}(n)$ non-parametric tabular imputation framework designed to solve the classic trade-off in missing data imputation:
* **As fast as Mean/Median** ($\mathcal{O}(n)$ linear time; scales effortlessly to millions of rows).
* **Neighborhood-sensitive like KNN** (captures local non-linear geometry without computing $\mathcal{O}(n^2)$ pairwise distance matrices).
* **Covariance-preserving like MICE** (restores multi-feature joint correlations in a single pass without iterative training loops).
* **Zero Mandatory Dependencies** (runs on 100% pure Python, but automatically accelerates and preserves types when Pandas or NumPy are present).

---

## 🚀 Installation

Install locally in editable mode:
```bash
pip install -e .
```

Or install with optional acceleration dependencies:
```bash
pip install -e ".[fast]"
```

---

## 💡 Quickstart

AMMI works seamlessly with **Pandas DataFrames**, **NumPy ndarrays**, and raw **Python Lists**:

### 1. With Pandas DataFrames
```python
import pandas as pd
import numpy as np
from ammi import AMMI

# Sample DataFrame with missing values
df = pd.DataFrame({
    'Age': [22, 25, np.nan, 40, 45, np.nan],
    'Experience': [1, 2, 2, 15, 20, 18],
    'Salary': [48000, 52000, np.nan, 120000, 140000, 135000]
})

# Impute in a single pass
imputer = AMMI(n_projections=4, n_bins=4, shrinkage_tau=3.0)
df_clean = imputer.fit_transform(df)

# Output is automatically returned as a Pandas DataFrame with original columns and index!
print(df_clean)
```

### 2. With Scikit-Learn Pipelines
AMMI is fully compatible with Scikit-Learn's `Pipeline` and `ColumnTransformer`:
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ammi import AMMI

pipeline = Pipeline([
    ('imputer', AMMI()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

### 3. Pure Python (No NumPy / Pandas required)
```python
from ammi import AMMI

data = [
    [22.0, 1.0, 48000.0],
    [25.0, 2.0, None],
    [None, 18.0, 135000.0]
]

imputer = AMMI()
clean_data = imputer.fit_transform(data)
```

---

## 🔬 Mathematical Formulation

AMMI is built on three core pillars:
1. **Orthonormal Random Slicing (ORS):** Projects continuous dimensions onto $L$ unit hyperplanes, partitioning the space into micro-neighborhoods in $\mathcal{O}(L \cdot d \cdot n)$ linear time.
2. **Empirical Bayes James-Stein Shrinkage:** Shrinks micro-cell expectations toward global robust medians as a function of cell sample density:
   $$\hat{x}_{\text{base}} = \left(\frac{N_{\text{cell}}}{N_{\text{cell}} + \tau}\right) \mu_{\text{local}} + \left(1 - \frac{N_{\text{cell}}}{N_{\text{cell}} + \tau}\right) \text{Median}_{\text{global}}$$
3. **Directed Covariance Residual Flow (DCRF):** Single-pass orthogonal residual projection that restores multivariate covariance without cyclic model fitting:
   $$\Delta_j = \left( \frac{\sum_{k \in \text{obs}} \rho_{jk} \cdot Z_k}{\sum_{k \in \text{obs}} |\rho_{jk}|} \right) \cdot \sigma_j$$

---

## 📖 Citation

```bibtex
@article{ammi2026,
  title   = {Adaptive Micro-Manifold Imputation: A Fast, Single-Pass Non-Parametric Framework for Tabular Missing Data},
  author  = {AMMI Authors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

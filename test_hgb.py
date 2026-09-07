from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd
import time

# Create 50,000 rows dataset
n_rows = 50000
rng = np.random.default_rng(42)

x1 = rng.normal(10, 2, n_rows)
x2 = x1 * 2.5 + rng.normal(0, 1, n_rows) # Linear correlation
x3 = x1**2 + rng.normal(0, 2, n_rows)    # Non-linear quadratic relationship

df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

# Introduce 10% missing values
mask = rng.uniform(0, 1, df.shape) < 0.1
df_missing = df.mask(mask)

print("Testing HistGradientBoosting in IterativeImputer...")
t0 = time.perf_counter()
imputer = IterativeImputer(
    estimator=HistGradientBoostingRegressor(max_iter=20, random_state=42),
    max_iter=3, # Fast convergence with trees
    random_state=42
)
res = imputer.fit_transform(df_missing)
t1 = time.perf_counter()
print(f"Success! Imputed {n_rows:,} rows in {t1 - t0:.2f} seconds.")

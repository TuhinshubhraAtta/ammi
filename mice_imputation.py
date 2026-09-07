import pandas as pd
import numpy as np

# Crucial step: IterativeImputer in scikit-learn requires enabling experimental feature
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

# 1. Create a dataset where multiple columns have missing values and cross-dependencies
# E.g., House Size (sqft), Bedrooms, and Price are correlated:
# Price ~ f(Size, Bedrooms)
# Size ~ f(Price, Bedrooms)
np.random.seed(42)

data = {
    'Size_sqft':   [1200, 1500, np.nan, 2400, 3000, np.nan, 1800, 2100, 3500],
    'Bedrooms':    [2,    3,    3,      np.nan, 4,    5,      3,    np.nan, 5],
    'Price_k':     [250,  310,  330,    480,    np.nan, 700,  370,  430,    720]
}

df = pd.DataFrame(data)

print("--- Original DataFrame with Multiple Missing Columns ---")
print(df)


# --- What is MICE (Chained Equations)? ---
print("\n" + "="*60)
print("--- How MICE (IterativeImputer) Works ---")
print("="*60)
print("""1. Initial step: Fill all NaNs temporarily (e.g. with mean).
2. Step 1: Set 'Size_sqft' as target y, and others as features X.
   Fit regression model on observed rows, then predict missing 'Size_sqft'.
3. Step 2: Set 'Bedrooms' as target y, use updated 'Size_sqft' + 'Price_k' as X.
   Fit regression model, then predict missing 'Bedrooms'.
4. Step 3: Set 'Price_k' as target y, use updated 'Size_sqft' + 'Bedrooms' as X.
   Fit regression model, then predict missing 'Price_k'.
5. Repeat steps 2-4 iteratively for 'max_iter' cycles until values stabilize!
""")


# --- Method 1: Default MICE using BayesianRidge Regressor ---
print("="*60)
print("--- 1. MICE with Bayesian Ridge (Linear Relationships) ---")
print("="*60)

mice_linear = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42,
    verbose=0
)

imputed_linear = mice_linear.fit_transform(df)
df_mice_linear = pd.DataFrame(imputed_linear, columns=df.columns)

# Round bedrooms to integer for realism
print("Imputed with Bayesian Ridge:")
print(df_mice_linear.round({'Size_sqft': 0, 'Bedrooms': 0, 'Price_k': 1}))


# --- Method 2: Non-Linear MICE with Random Forest ---
print("\n" + "="*60)
print("--- 2. MICE with Random Forest (Non-Linear & Complex Data) ---")
print("="*60)

mice_rf = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=42
)

imputed_rf = mice_rf.fit_transform(df)
df_mice_rf = pd.DataFrame(imputed_rf, columns=df.columns)

print("Imputed with Random Forest:")
print(df_mice_rf.round({'Size_sqft': 0, 'Bedrooms': 0, 'Price_k': 1}))


# --- Method 3: Multiple Imputations (Capturing Uncertainty) ---
print("\n" + "="*60)
print("--- 3. Generating Multiple Datasets (m=3) with Posterior Sampling ---")
print("="*60)
print("True statistical MICE generates 'm' different completions by sampling from")
print("the predictive distribution, allowing uncertainty estimation:\n")

for i in range(3):
    mice_sampler = IterativeImputer(
        sample_posterior=True,  # Draws from predictive distribution to model uncertainty
        random_state=i + 1,
        max_iter=10
    )
    sampled_df = pd.DataFrame(mice_sampler.fit_transform(df), columns=df.columns)
    
    # Let's inspect Row 4 (missing Price_k) and Row 2 (missing Size_sqft) across imputations
    print(f"Dataset {i+1}:")
    print(f"  Row 2 Imputed Size:  {sampled_df.loc[2, 'Size_sqft']:.0f} sqft")
    print(f"  Row 4 Imputed Price: ${sampled_df.loc[4, 'Price_k']:.1f}k")

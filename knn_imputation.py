import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# 1. Create a dataset with correlated features
# Notice how Experience, Age, and Salary are correlated:
# - Junior profile: Age ~22-25, Experience ~1-2, Salary ~50k
# - Senior profile: Age ~40-45, Experience ~15-20, Salary ~120k-140k
data = {
    'Age': [22, 24, 25, 40, 42, 45, np.nan],       # Row 6 has missing Age
    'Experience': [1, 2, 2, 15, 17, 20, 18],       # Row 6 has 18 years exp (clearly a senior!)
    'Salary': [48000, 52000, np.nan, 120000, 128000, 140000, 135000] # Row 2 has missing Salary
}

df = pd.DataFrame(data)

print("--- Original DataFrame with Missing Values ---")
print(df)
print("\nLook at Row 6: Experience is 18, Salary is $135,000, but Age is NaN.")
print("Look at Row 2: Age is 25, Experience is 2, but Salary is NaN.")


# --- Method 1: Basic KNNImputer ---
print("\n" + "="*50)
print("--- 1. Basic KNNImputer (n_neighbors=2) ---")
print("="*50)

# Initialize KNNImputer looking for 2 nearest neighbors
imputer_knn = KNNImputer(n_neighbors=2)
imputed_array = imputer_knn.fit_transform(df)

df_knn = pd.DataFrame(imputed_array, columns=df.columns)
print(df_knn)


# --- Compare with Mean / Median Imputer ---
print("\n" + "="*50)
print("--- Comparison: KNN vs Mean Imputation ---")
print("="*50)

imputer_mean = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(imputer_mean.fit_transform(df), columns=df.columns)

print(f"Mean Imputed Age for Row 6 (Senior with 18 yrs exp):   {df_mean.loc[6, 'Age']:.1f} years  <-- Unrealistic (too young)")
print(f"KNN Imputed Age for Row 6 (Senior with 18 yrs exp):    {df_knn.loc[6, 'Age']:.1f} years  <-- Matches senior peers!")

print(f"\nMean Imputed Salary for Row 2 (Junior with 2 yrs exp): ${df_mean.loc[2, 'Salary']:,.0f}  <-- Inflated by senior salaries")
print(f"KNN Imputed Salary for Row 2 (Junior with 2 yrs exp):  ${df_knn.loc[2, 'Salary']:,.0f}  <-- Matches junior peers!")


# --- Method 2: Scaled KNNImputer (Best Practice) ---
print("\n" + "="*50)
print("--- 2. Scaled KNNImputer (Best Practice) ---")
print("="*50)
print("Why Scale? KNN uses Euclidean distance. If Salary is ~100,000 and Age is ~30,")
print("Salary differences will overpower Age differences unless scaled between 0 and 1.\n")

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

imputer_scaled_knn = KNNImputer(n_neighbors=2, weights='distance') # distance-weighted
df_imputed_scaled = imputer_scaled_knn.fit_transform(df_scaled)

# Inverse transform back to original scale
df_final = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns)
print("Final DataFrame after Scaled & Weighted KNN Imputation:")
print(df_final.round({'Age': 1, 'Experience': 1, 'Salary': 0}))

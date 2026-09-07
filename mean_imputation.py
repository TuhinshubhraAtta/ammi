import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 1. Create a sample dataset with missing values
print("--- Mean Imputation Example ---")
data = {
    'Age': [25, 30, np.nan, 35, 40, np.nan, 50],
    'Salary': [50000, 60000, 65000, np.nan, 80000, 85000, np.nan],
    'City': ['New York', 'London', 'Paris', 'New York', np.nan, 'London', 'Paris'] # Mean imputation doesn't work on categorical columns
}

df = pd.DataFrame(data)
print("\nOriginal DataFrame with missing values:")
print(df)

# --- Method 1: Using Pandas ---
print("\n\n--- Method 1: Using Pandas .fillna() ---")
df_pandas_imputed = df.copy()

# Calculate means
age_mean = df_pandas_imputed['Age'].mean()
salary_mean = df_pandas_imputed['Salary'].mean()

print(f"Calculated Mean for Age: {age_mean:.2f}")
print(f"Calculated Mean for Salary: {salary_mean:.2f}")

# Fill missing values
df_pandas_imputed['Age'] = df_pandas_imputed['Age'].fillna(age_mean)
df_pandas_imputed['Salary'] = df_pandas_imputed['Salary'].fillna(salary_mean)

print("\nDataFrame after Pandas Mean Imputation:")
print(df_pandas_imputed)


# --- Method 2: Using Scikit-Learn SimpleImputer ---
print("\n\n--- Method 2: Using Scikit-Learn SimpleImputer ---")
df_sklearn_imputed = df.copy()

# Initialize the imputer (strategy='mean' is the default)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit and transform the numerical columns
# We only select numerical columns because mean imputation is not defined for strings
numerical_cols = ['Age', 'Salary']
imputed_values = imputer.fit_transform(df_sklearn_imputed[numerical_cols])

# Assign the imputed values back to the dataframe
df_sklearn_imputed[numerical_cols] = imputed_values

print(f"Scikit-Learn calculated means: {imputer.statistics_}")

print("\nDataFrame after Scikit-Learn Mean Imputation:")
print(df_sklearn_imputed)

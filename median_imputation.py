import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 1. Create a sample dataset with missing values and an outlier
# Notice the outlier in Salary (1,000,000) which would heavily distort the mean!
data = {
    'Age': [22, 25, np.nan, 28, 29, 32, np.nan, 45],
    'Salary': [45000, 50000, 52000, np.nan, 55000, 60000, np.nan, 1000000],  # Outlier: 1,000,000
    'Department': ['IT', 'HR', 'IT', 'Finance', np.nan, 'HR', 'IT', 'Executive']
}

df = pd.DataFrame(data)

print("--- Original DataFrame (Note the Salary outlier: 1,000,000) ---")
print(df)

# Compare Mean vs Median to demonstrate why Median is chosen
mean_salary = df['Salary'].mean()
median_salary = df['Salary'].median()

print("\n--- Why Median Imputation? ---")
print(f"Mean Salary:   ${mean_salary:,.2f}  <-- Skewed heavily by the $1,000,000 outlier")
print(f"Median Salary: ${median_salary:,.2f}  <-- Robust and representative of typical values")


# --- Method 1: Using Pandas .fillna() with .median() ---
print("\n\n--- Method 1: Using Pandas .fillna() ---")
df_pandas_imputed = df.copy()

# Calculate medians for numeric columns
age_median = df_pandas_imputed['Age'].median()
salary_median = df_pandas_imputed['Salary'].median()

print(f"Calculated Median for Age:    {age_median:.1f}")
print(f"Calculated Median for Salary: {salary_median:,.1f}")

# Impute missing values
df_pandas_imputed['Age'] = df_pandas_imputed['Age'].fillna(age_median)
df_pandas_imputed['Salary'] = df_pandas_imputed['Salary'].fillna(salary_median)

print("\nDataFrame after Pandas Median Imputation:")
print(df_pandas_imputed)


# --- Method 2: Using Scikit-Learn SimpleImputer(strategy='median') ---
print("\n\n--- Method 2: Using Scikit-Learn SimpleImputer ---")
df_sklearn_imputed = df.copy()

# Initialize SimpleImputer with strategy='median'
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# Fit and transform numerical columns
numerical_cols = ['Age', 'Salary']
df_sklearn_imputed[numerical_cols] = imputer.fit_transform(df_sklearn_imputed[numerical_cols])

print(f"Scikit-Learn calculated statistics (medians): {imputer.statistics_}")

print("\nDataFrame after Scikit-Learn Median Imputation:")
print(df_sklearn_imputed)

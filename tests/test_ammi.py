import unittest
import math
from ammi import AMMI, AdaptiveMicroManifoldImputer

class TestAMMI(unittest.TestCase):
    def setUp(self):
        self.raw_data = [
            [22.0, 1.0, 48000.0],
            [24.0, 2.0, 52000.0],
            [25.0, 2.0, None],              # Missing Salary
            [40.0, 15.0, 120000.0],
            [42.0, 17.0, 128000.0],
            [45.0, 20.0, 140000.0],
            [None, 18.0, 135000.0],         # Missing Age
        ]

    def test_pure_python_list(self):
        imputer = AMMI(n_projections=3, n_bins=3, shrinkage_tau=2.0, seed=42)
        clean = imputer.fit_transform(self.raw_data)
        
        self.assertEqual(len(clean), len(self.raw_data))
        self.assertEqual(len(clean[0]), len(self.raw_data[0]))
        
        # Verify no missing values remaining
        for row in clean:
            for val in row:
                self.assertIsNotNone(val)
                self.assertFalse(math.isnan(val))

        # Check values are reasonable
        junior_salary = clean[2][2]
        senior_age = clean[6][0]
        self.assertTrue(40000.0 <= junior_salary <= 90000.0)
        self.assertTrue(30.0 <= senior_age <= 55.0)

    def test_pandas_dataframe(self):
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            self.skipTest("Pandas/NumPy not installed")

        df = pd.DataFrame(self.raw_data, columns=['Age', 'Experience', 'Salary'])
        imputer = AMMI(n_projections=3, n_bins=3, shrinkage_tau=2.0, seed=42)
        clean_df = imputer.fit_transform(df)

        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertEqual(list(clean_df.columns), ['Age', 'Experience', 'Salary'])
        self.assertEqual(clean_df.isna().sum().sum(), 0)

    def test_numpy_array(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed")

        arr = np.array([
            [1.0, 2.0, np.nan],
            [2.0, 3.0, 6.0],
            [np.nan, 4.0, 8.0]
        ])
        imputer = AMMI(n_projections=2, n_bins=2, seed=42)
        clean_arr = imputer.fit_transform(arr)

        self.assertIsInstance(clean_arr, np.ndarray)
        self.assertFalse(np.isnan(clean_arr).any())

if __name__ == '__main__':
    unittest.main()

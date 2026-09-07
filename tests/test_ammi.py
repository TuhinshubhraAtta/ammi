import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError

from ammi import AMMI, AdaptiveMicroManifoldImputer


class TestAdaptiveMicroManifoldImputer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Generate correlated multivariate Gaussian data
        mean = np.array([10.0, 20.0, 50.0, 100.0])
        cov = np.array([
            [4.0, 2.5, -1.8, 3.0],
            [2.5, 9.0, -2.0, 4.5],
            [-1.8, -2.0, 16.0, -5.0],
            [3.0, 4.5, -5.0, 25.0],
        ])
        self.X_clean = self.rng.multivariate_normal(mean, cov, size=500)

        # Introduce 15% missing completely at random (MCAR)
        self.mask = self.rng.uniform(0.0, 1.0, size=self.X_clean.shape) < 0.15
        self.X_missing = self.X_clean.copy()
        self.X_missing[self.mask] = np.nan

    def test_basic_imputation(self):
        """Verify all missing values are imputed with finite numbers."""
        imputer = AMMI(n_projections=3, n_bins=4, random_state=42)
        X_imp = imputer.fit_transform(self.X_missing)

        self.assertEqual(X_imp.shape, self.X_missing.shape)
        self.assertFalse(np.isnan(X_imp).any())
        self.assertTrue(np.isfinite(X_imp).all())

    def test_covariance_recovery(self):
        """Verify correlation matrix of imputed data closely recovers true covariance."""
        imputer = AMMI(n_projections=4, n_bins=4, shrinkage_tau=2.0, random_state=42)
        X_imp = imputer.fit_transform(self.X_missing)

        true_corr = np.corrcoef(self.X_clean, rowvar=False)
        imp_corr = np.corrcoef(X_imp, rowvar=False)

        # Correlation distortion (mean absolute difference across off-diagonal elements)
        off_diag = ~np.eye(4, dtype=bool)
        distortion = np.mean(np.abs(true_corr[off_diag] - imp_corr[off_diag]))

        # Should exhibit minimal correlation distortion (< 0.08 on this sample)
        self.assertLess(distortion, 0.08)

    def test_pandas_preservation(self):
        """Verify DataFrame columns, indices, and type are preserved."""
        cols = ["feat_A", "feat_B", "feat_C", "feat_D"]
        idx = [f"sample_{i}" for i in range(len(self.X_missing))]
        df_missing = pd.DataFrame(self.X_missing, index=idx, columns=cols)

        imputer = AMMI(n_projections=3, n_bins=3, random_state=42)
        df_imp = imputer.fit_transform(df_missing)

        self.assertIsInstance(df_imp, pd.DataFrame)
        self.assertEqual(list(df_imp.columns), cols)
        self.assertEqual(list(df_imp.index), idx)
        self.assertEqual(df_imp.isna().sum().sum(), 0)

    def test_numpy_preservation(self):
        """Verify NumPy array input returns NumPy array output."""
        imputer = AMMI(n_projections=2, n_bins=2, random_state=42)
        arr_imp = imputer.fit_transform(self.X_missing)

        self.assertIsInstance(arr_imp, np.ndarray)
        self.assertEqual(arr_imp.dtype, np.float64)
        self.assertEqual(arr_imp.shape, self.X_missing.shape)

    def test_list_preservation(self):
        """Verify pure Python nested list input returns nested lists."""
        raw_list = [
            [1.0, 2.0, None],
            [2.0, None, 6.0],
            [3.0, 4.0, 9.0],
            [4.0, 5.0, 12.0],
        ]
        imputer = AMMI(n_projections=2, n_bins=2, random_state=42)
        clean_list = imputer.fit_transform(raw_list)

        self.assertIsInstance(clean_list, list)
        self.assertIsInstance(clean_list[0], list)
        self.assertEqual(len(clean_list), 4)
        self.assertEqual(len(clean_list[0]), 3)

    def test_sklearn_pipeline(self):
        """Verify full compatibility within a scikit-learn Pipeline."""
        y = self.X_clean[:, 0] * 1.5 - self.X_clean[:, 1] * 0.8 + self.rng.normal(0, 0.1, len(self.X_clean))
        pipe = Pipeline([
            ("imputer", AMMI(n_projections=3, n_bins=3, random_state=42)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ])

        pipe.fit(self.X_missing, y)
        predictions = pipe.predict(self.X_missing)
        self.assertEqual(len(predictions), len(y))
        self.assertFalse(np.isnan(predictions).any())

    def test_reproducibility(self):
        """Verify deterministic behavior with fixed random_state."""
        imp1 = AMMI(n_projections=3, n_bins=3, random_state=123)
        res1 = imp1.fit_transform(self.X_missing)

        imp2 = AMMI(n_projections=3, n_bins=3, random_state=123)
        res2 = imp2.fit_transform(self.X_missing)

        np.testing.assert_allclose(res1, res2)

    def test_edge_cases(self):
        """Test edge cases: no NaNs, constant column, not fitted error."""
        imputer = AMMI()

        # NotFittedError before fit
        with self.assertRaises(NotFittedError):
            imputer.transform(self.X_clean)

        # No NaNs fast path
        X_no_nan = np.array([[1.0, 2.0], [3.0, 4.0]])
        imputed_no_nan = imputer.fit_transform(X_no_nan)
        np.testing.assert_array_equal(X_no_nan, imputed_no_nan)

        # Constant feature column (zero standard deviation)
        X_const = np.array([
            [5.0, 1.0],
            [5.0, np.nan],
            [5.0, 3.0],
        ])
        imputed_const = imputer.fit_transform(X_const)
        self.assertFalse(np.isnan(imputed_const).any())
        self.assertTrue(np.all(imputed_const[:, 0] == 5.0))

        # Empty matrix error
        with self.assertRaises(ValueError):
            imputer.fit(np.empty((0, 2)))

        # Feature count mismatch in transform
        imputer.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with self.assertRaises(ValueError):
            imputer.transform(np.array([[1.0, 2.0, 3.0]]))


if __name__ == "__main__":
    unittest.main()

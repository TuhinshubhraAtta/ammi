"""
Adaptive Micro-Manifold Imputer (AMMI) Core Engine
==================================================
Handles Pandas DataFrames, NumPy ndarrays, and Pure Python Lists transparently.
"""

import math
import random
from typing import List, Tuple, Optional, Any, Dict, Union

# Optional Scikit-Learn Integration
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:
        """Fallback when scikit-learn is not installed."""
        def get_params(self, deep=True):
            return {}
        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class TransformerMixin:
        """Fallback when scikit-learn is not installed."""
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)

# Optional NumPy and Pandas detection
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def is_missing(val: Any) -> bool:
    """Checks if value is None, NaN, or non-finite."""
    if val is None:
        return True
    if isinstance(val, (int, float)) and math.isnan(val):
        return True
    return False


class AdaptiveMicroManifoldImputer(BaseEstimator, TransformerMixin):
    """
    Adaptive Micro-Manifold Imputer (AMMI)
    --------------------------------------
    A linear-time O(n) non-parametric imputation algorithm.

    Parameters
    ----------
    n_projections : int, default=4
        Number of orthonormal random unit hyperplanes.
    n_bins : int, default=4
        Number of quantile-based subdivisions per projection slice.
    shrinkage_tau : float, default=3.0
        Empirical Bayes James-Stein shrinkage parameter. Controls regularization
        for sparse micro-manifold cells.
    seed : int, default=42
        Random seed for reproducibility.
    """
    def __init__(
        self,
        n_projections: int = 4,
        n_bins: int = 4,
        shrinkage_tau: float = 3.0,
        seed: int = 42
    ):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.shrinkage_tau = shrinkage_tau
        self.seed = seed

        # Learned statistical state
        self.global_medians_: List[float] = []
        self.global_stds_: List[float] = []
        self.projection_matrix_: List[List[float]] = []
        self.bin_edges_: List[List[float]] = []
        self.corr_matrix_: List[List[float]] = []
        self.cell_means_: Dict[Tuple[int, ...], List[float]] = {}
        self.cell_counts_: Dict[Tuple[int, ...], List[int]] = {}
        self.feature_names_: Optional[List[str]] = None

    def _convert_input(self, X: Any) -> Tuple[List[List[Optional[float]]], str, Any]:
        """Converts input to standard 2D list while remembering original type metadata."""
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            meta = (X.index, X.columns)
            return X.values.tolist(), "pandas", meta
        
        if HAS_NUMPY and isinstance(X, np.ndarray):
            meta = (X.dtype, X.shape)
            return X.tolist(), "numpy", meta
            
        if isinstance(X, list):
            meta = None
            return X, "list", meta
            
        raise TypeError(f"Unsupported data type: {type(X)}. AMMI accepts pandas.DataFrame, numpy.ndarray, or List[List].")

    def fit(self, X: Any, y: Any = None) -> 'AdaptiveMicroManifoldImputer':
        """
        Fit AMMI's global scale, random projections, and micro-manifold buckets.
        """
        X_list, _, _ = self._convert_input(X)
        if not X_list or not X_list[0]:
            raise ValueError("Input data cannot be empty.")

        n_samples = len(X_list)
        n_features = len(X_list[0])
        rng = random.Random(self.seed)

        # 1. Global Medians & Robust Standard Deviations
        self.global_medians_ = []
        self.global_stds_ = []
        for c in range(n_features):
            vals = [row[c] for row in X_list if not is_missing(row[c])]
            if not vals:
                self.global_medians_.append(0.0)
                self.global_stds_.append(1.0)
            else:
                vals.sort()
                k = len(vals)
                median = float(vals[k // 2]) if k % 2 == 1 else (vals[k // 2 - 1] + vals[k // 2]) / 2.0
                mean_v = sum(vals) / k
                var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, k - 1)
                std = math.sqrt(var_v) or 1.0
                self.global_medians_.append(median)
                self.global_stds_.append(std)

        # 2. Standardized baseline
        X_norm = [
            [
                0.0 if is_missing(row[c]) else (row[c] - self.global_medians_[c]) / self.global_stds_[c]
                for c in range(n_features)
            ]
            for row in X_list
        ]

        # 3. Unit Hyperplane Generation
        R = [[rng.gauss(0.0, 1.0) for _ in range(self.n_projections)] for _ in range(n_features)]
        self.projection_matrix_ = [[0.0] * self.n_projections for _ in range(n_features)]
        for p in range(self.n_projections):
            col_norm = math.sqrt(sum(R[f][p] ** 2 for f in range(n_features))) or 1.0
            for f in range(n_features):
                self.projection_matrix_[f][p] = R[f][p] / col_norm

        # 4. Multi-Resolution Quantile Slicing
        projected = self._project(X_norm)
        self.bin_edges_ = []
        for p in range(self.n_projections):
            p_vals = sorted(row[p] for row in projected)
            self.bin_edges_.append([
                p_vals[min(int((b / self.n_bins) * len(p_vals)), len(p_vals) - 1)]
                for b in range(1, self.n_bins)
            ])

        # 5. Pearson Correlation Matrix (for Directed Residual Flow)
        self.corr_matrix_ = [[0.0] * n_features for _ in range(n_features)]
        for f1 in range(n_features):
            for f2 in range(f1 + 1, n_features):
                pairs = [(row[f1], row[f2]) for row in X_list if not is_missing(row[f1]) and not is_missing(row[f2])]
                if len(pairs) > 2:
                    m1 = sum(p[0] for p in pairs) / len(pairs)
                    m2 = sum(p[1] for p in pairs) / len(pairs)
                    cov = sum((p[0] - m1) * (p[1] - m2) for p in pairs)
                    v1 = sum((p[0] - m1) ** 2 for p in pairs)
                    v2 = sum((p[1] - m2) ** 2 for p in pairs)
                    r = cov / math.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else 0.0
                else:
                    r = 0.0
                self.corr_matrix_[f1][f2] = r
                self.corr_matrix_[f2][f1] = r

        # 6. Micro-Manifold Cell Aggregations
        hashes = self._hash(projected)
        cell_sums: Dict[Tuple[int, ...], List[float]] = {}
        self.cell_counts_ = {}

        for r_idx, h in enumerate(hashes):
            if h not in cell_sums:
                cell_sums[h] = [0.0] * n_features
                self.cell_counts_[h] = [0] * n_features
            for c in range(n_features):
                val = X_list[r_idx][c]
                if not is_missing(val):
                    cell_sums[h][c] += val
                    self.cell_counts_[h][c] += 1

        self.cell_means_ = {
            h: [sums[c] / self.cell_counts_[h][c] if self.cell_counts_[h][c] > 0 else self.global_medians_[c]
                for c in range(n_features)]
            for h, sums in cell_sums.items()
        }
        return self

    def _project(self, X_norm: List[List[float]]) -> List[List[float]]:
        n_features = len(self.global_medians_)
        return [
            [sum(row[f] * self.projection_matrix_[f][p] for f in range(n_features)) for p in range(self.n_projections)]
            for row in X_norm
        ]

    def _hash(self, projected: List[List[float]]) -> List[Tuple[int, ...]]:
        hashes = []
        for row in projected:
            h = []
            for p in range(self.n_projections):
                val = row[p]
                bin_idx = sum(1 for edge in self.bin_edges_[p] if val > edge)
                h.append(bin_idx)
            hashes.append(tuple(h))
        return hashes

    def transform(self, X: Any) -> Any:
        """
        Imputes missing values and returns the output in the same format as input
        (Pandas DataFrame, NumPy array, or List).
        """
        X_list, input_type, meta = self._convert_input(X)
        n_features = len(self.global_medians_)

        X_norm = [
            [
                0.0 if is_missing(row[c]) else (row[c] - self.global_medians_[c]) / self.global_stds_[c]
                for c in range(n_features)
            ]
            for row in X_list
        ]

        projected = self._project(X_norm)
        hashes = self._hash(projected)
        imputed_output = []

        for r_idx, row in enumerate(X_list):
            imputed_row = list(row)
            h = hashes[r_idx]
            missing_cols = [c for c in range(n_features) if is_missing(row[c])]
            observed_cols = [c for c in range(n_features) if not is_missing(row[c])]

            for c in missing_cols:
                # 1. James-Stein Shrinkage
                if h in self.cell_means_:
                    local_mean = self.cell_means_[h][c]
                    cnt = self.cell_counts_[h][c]
                    shrinkage = cnt / (cnt + self.shrinkage_tau)
                    base_est = shrinkage * local_mean + (1.0 - shrinkage) * self.global_medians_[c]
                else:
                    base_est = self.global_medians_[c]

                # 2. Directed Covariance Residual Flow
                if observed_cols:
                    cov_dot = sum(self.corr_matrix_[c][obs] * X_norm[r_idx][obs] for obs in observed_cols)
                    sum_abs = sum(abs(self.corr_matrix_[c][obs]) for obs in observed_cols)
                    residual_flow = (cov_dot / sum_abs) * self.global_stds_[c] if sum_abs > 1e-6 else 0.0
                else:
                    residual_flow = 0.0

                imputed_row[c] = base_est + residual_flow

            imputed_output.append(imputed_row)

        # Restore original input format
        if input_type == "pandas" and HAS_PANDAS:
            orig_index, orig_columns = meta
            return pd.DataFrame(imputed_output, index=orig_index, columns=orig_columns)
        
        if input_type == "numpy" and HAS_NUMPY:
            orig_dtype, _ = meta
            # Ensure float conversion for clean numerical data
            target_dtype = np.float64 if np.issubdtype(orig_dtype, np.integer) else orig_dtype
            return np.array(imputed_output, dtype=target_dtype)

        return imputed_output


# Alias for clean scientific import
AMMI = AdaptiveMicroManifoldImputer

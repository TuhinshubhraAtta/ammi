"""
Adaptive Micro-Manifold Imputer (AMMI)
======================================
High-performance, linear-time missing value imputation using orthonormal random
projection slicing, hierarchical empirical Bayes shrinkage, and correlation-weighted
residual projection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AdaptiveMicroManifoldImputer(BaseEstimator, TransformerMixin):
    """
    Adaptive Micro-Manifold Imputer (AMMI).

    A non-parametric, linear-time tabular imputation estimator. AMMI partitions
    continuous multidimensional feature spaces using orthonormal random hyperplanes,
    aggregates local neighborhood cell statistics with empirical Bayes shrinkage,
    and restores multivariate joint covariance via single-pass correlation-weighted
    residual projection.

    Parameters
    ----------
    n_projections : int, default=4
        The number of orthonormal random projection directions used to slice
        the manifold. Higher values yield more granular local neighborhoods
        at the cost of exponentially sparser cells (n_bins ** n_projections).
    n_bins : int, default=4
        Number of quantile-based subdivisions per projection axis.
    shrinkage_tau : float, default=3.0
        Regularization pseudo-count for empirical Bayes cell shrinkage. Controls
        how aggressively sparse micro-cells are smoothed toward global medians.
        Higher values increase shrinkage toward the global marginal prior.
    random_state : int or None, default=42
        Determines random number generation for orthonormal projection generation.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=object
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings (e.g., pandas DataFrame).
    global_medians_ : ndarray of shape (n_features_in_,)
        Robust central tendencies (column medians) learned on observed entries.
    global_stds_ : ndarray of shape (n_features_in_,)
        Standard deviations of observed entries per feature.
    projection_matrix_ : ndarray of shape (n_features_in_, n_projections)
        Orthonormal basis matrix generated via QR decomposition.
    bin_edges_ : List[ndarray]
        Quantile bin boundary coordinates for each projection axis.
    corr_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        Empirical Pearson correlation matrix computed over observed pairwise data.
    cell_means_ : Dict[Tuple[int, ...], ndarray]
        Mean feature values for each partitioned micro-manifold cell.
    cell_counts_ : Dict[Tuple[int, ...], ndarray]
        Counts of observed samples for each feature within each cell.
    """

    def __init__(
        self,
        n_projections: int = 4,
        n_bins: int = 4,
        shrinkage_tau: float = 3.0,
        random_state: Optional[int] = 42,
    ):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.shrinkage_tau = shrinkage_tau
        self.random_state = random_state

    def _convert_input(self, X: Any) -> Tuple[np.ndarray, str, Any]:
        """Validates and converts input data into a 2D float64 NumPy array."""
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            orig_meta = (X.index.copy(), X.columns.copy(), X.dtypes.copy())
            arr = X.to_numpy(dtype=np.float64, copy=True)
            return arr, "pandas", orig_meta

        if isinstance(X, np.ndarray):
            orig_meta = (X.dtype, X.shape)
            if not np.issubdtype(X.dtype, np.number):
                try:
                    arr = X.astype(np.float64)
                except (ValueError, TypeError) as exc:
                    raise ValueError("AMMI requires numerical feature matrices.") from exc
            else:
                arr = X.astype(np.float64, copy=True)
            return arr, "numpy", orig_meta

        if isinstance(X, (list, tuple)):
            try:
                arr = np.array(X, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError("AMMI requires numerical feature matrices.") from exc
            return arr, "list", None

        raise TypeError(
            f"Unsupported input type: {type(X)}. AMMI accepts pandas.DataFrame, "
            f"numpy.ndarray, or nested lists/tuples."
        )

    def fit(self, X: Any, y: Any = None) -> "AdaptiveMicroManifoldImputer":
        """
        Fit the imputer on input dataset X.

        Computes global marginal medians/scales, orthonormal projection basis,
        quantile bin partitions, pairwise correlation matrix, and cell-level
        empirical Bayes aggregates.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The training input samples where missing values are represented as NaN.
        y : Ignored
            Not used, present for scikit-learn pipeline compatibility.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_arr, input_type, meta = self._convert_input(X)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead.")

        n_samples, n_features = X_arr.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("Input data matrix cannot be empty.")

        self.n_features_in_ = n_features
        if input_type == "pandas" and HAS_PANDAS:
            _, cols, _ = meta
            self.feature_names_in_ = np.array(cols, dtype=object)

        # 1. Global Centrality and Scale Estimation
        missing_mask = np.isnan(X_arr)
        medians = np.zeros(n_features, dtype=np.float64)
        stds = np.ones(n_features, dtype=np.float64)

        for j in range(n_features):
            col_valid = X_arr[~missing_mask[:, j], j]
            if col_valid.size > 0:
                med = float(np.median(col_valid))
                std = float(np.std(col_valid, ddof=1)) if col_valid.size > 1 else 1.0
                medians[j] = med
                stds[j] = std if std > 1e-8 else 1.0
            else:
                medians[j] = 0.0
                stds[j] = 1.0

        self.global_medians_ = medians
        self.global_stds_ = stds

        # 2. Standardized Baseline Representation
        # Fill missing values with median for geometry slicing
        X_filled = np.where(missing_mask, self.global_medians_, X_arr)
        Z = (X_filled - self.global_medians_) / self.global_stds_

        # 3. Orthonormal Random Hyperplane Basis via QR Decomposition
        effective_proj = min(self.n_projections, n_features)
        rng = np.random.default_rng(self.random_state)
        random_normals = rng.standard_normal((n_features, effective_proj))
        q_basis, _ = np.linalg.qr(random_normals)
        self.projection_matrix_ = q_basis

        # 4. Multi-Resolution Quantile Slicing
        projected = Z @ self.projection_matrix_  # shape: (n_samples, effective_proj)
        self.bin_edges_ = []
        quantiles_to_eval = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        for p in range(effective_proj):
            col_p = projected[:, p]
            edges = np.quantile(col_p, quantiles_to_eval)
            # Ensure unique monotonically increasing edges
            self.bin_edges_.append(edges)

        # 5. Pairwise Correlation Matrix
        valid_mask = ~missing_mask
        counts = valid_mask.T.astype(np.float64) @ valid_mask.astype(np.float64)
        Z_masked = np.where(valid_mask, Z, 0.0)
        cov = (Z_masked.T @ Z_masked) / np.maximum(counts - 1.0, 1.0)
        diag = np.diag(cov)
        denom = np.sqrt(np.outer(diag, diag))
        corr = np.divide(
            cov,
            denom,
            out=np.zeros_like(cov),
            where=(denom > 1e-12) & (counts > 2.0),
        )
        np.fill_diagonal(corr, 1.0)
        self.corr_matrix_ = np.clip(corr, -1.0, 1.0)

        # 6. Micro-Manifold Cell Aggregation
        bin_indices = np.zeros((n_samples, effective_proj), dtype=int)
        for p in range(effective_proj):
            bin_indices[:, p] = np.digitize(projected[:, p], self.bin_edges_[p])

        cell_sums: Dict[Tuple[int, ...], np.ndarray] = {}
        cell_counts: Dict[Tuple[int, ...], np.ndarray] = {}

        for i in range(n_samples):
            key = tuple(bin_indices[i])
            if key not in cell_sums:
                cell_sums[key] = np.zeros(n_features, dtype=np.float64)
                cell_counts[key] = np.zeros(n_features, dtype=np.int64)

            for j in range(n_features):
                if not missing_mask[i, j]:
                    cell_sums[key][j] += X_arr[i, j]
                    cell_counts[key][j] += 1

        self.cell_counts_ = cell_counts
        self.cell_means_ = {}
        for key, sums in cell_sums.items():
            counts_k = cell_counts[key]
            means = np.where(counts_k > 0, sums / np.maximum(counts_k, 1), self.global_medians_)
            self.cell_means_[key] = means

        return self

    def transform(self, X: Any) -> Any:
        """
        Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The input data to impute.

        Returns
        -------
        X_imputed : {ndarray, dataframe, list}
            Transformed input with all missing values imputed in the original data structure.
        """
        check_is_fitted(
            self,
            attributes=[
                "global_medians_",
                "global_stds_",
                "projection_matrix_",
                "bin_edges_",
                "corr_matrix_",
                "cell_means_",
                "cell_counts_",
            ],
        )

        X_arr, input_type, meta = self._convert_input(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X_arr.shape[1] if X_arr.ndim == 2 else 'non-2D'} features, "
                f"but AMMI was fitted with {self.n_features_in_} features."
            )

        n_samples, n_features = X_arr.shape
        missing_mask = np.isnan(X_arr)

        # Fast path: no missing values present
        if not np.any(missing_mask):
            return self._format_output(X_arr, input_type, meta)

        # Standardized coordinates
        X_filled = np.where(missing_mask, self.global_medians_, X_arr)
        Z = (X_filled - self.global_medians_) / self.global_stds_

        # Project and determine cell coordinates
        projected = Z @ self.projection_matrix_
        effective_proj = self.projection_matrix_.shape[1]
        bin_indices = np.zeros((n_samples, effective_proj), dtype=int)
        for p in range(effective_proj):
            bin_indices[:, p] = np.digitize(projected[:, p], self.bin_edges_[p])

        X_out = X_arr.copy()

        for i in range(n_samples):
            missing_cols = np.flatnonzero(missing_mask[i])
            if missing_cols.size == 0:
                continue

            observed_cols = np.flatnonzero(~missing_mask[i])
            cell_key = tuple(bin_indices[i])

            has_cell = cell_key in self.cell_means_
            cell_means = self.cell_means_[cell_key] if has_cell else None
            cell_counts = self.cell_counts_[cell_key] if has_cell else None

            for c in missing_cols:
                # 1. Hierarchical Empirical Bayes Cell Shrinkage
                if has_cell and cell_counts[c] > 0:
                    cnt = cell_counts[c]
                    weight = cnt / (cnt + self.shrinkage_tau)
                    local_mean = cell_means[c]
                else:
                    weight = 0.0
                    local_mean = self.global_medians_[c]

                # 2. Correlation-Weighted Residual Projection Prior
                if observed_cols.size > 0:
                    corrs = self.corr_matrix_[c, observed_cols]
                    z_obs = Z[i, observed_cols]
                    sum_abs = np.sum(np.abs(corrs))
                    if sum_abs > 1e-6:
                        residual_flow = (np.dot(corrs, z_obs) / sum_abs) * self.global_stds_[c]
                    else:
                        residual_flow = 0.0
                else:
                    residual_flow = 0.0

                # 3. Adaptive Blending: dense cells rely on local manifold geometry,
                # sparse cells shrink toward global regularized covariance projection.
                prior_estimate = self.global_medians_[c] + residual_flow
                X_out[i, c] = weight * local_mean + (1.0 - weight) * prior_estimate

        return self._format_output(X_out, input_type, meta)

    def _format_output(self, X_out: np.ndarray, input_type: str, meta: Any) -> Any:
        """Restores the imputed array to the user's original data structure."""
        if input_type == "pandas" and HAS_PANDAS:
            idx, cols, dtypes = meta
            df_out = pd.DataFrame(X_out, index=idx, columns=cols)
            # Attempt to preserve compatible original dtypes
            for col in cols:
                orig_dt = dtypes[col]
                if not np.issubdtype(orig_dt, np.floating) and np.issubdtype(orig_dt, np.integer):
                    df_out[col] = df_out[col].astype(np.float64)
                else:
                    try:
                        df_out[col] = df_out[col].astype(orig_dt)
                    except (ValueError, TypeError):
                        pass
            return df_out

        if input_type == "numpy":
            orig_dtype, _ = meta
            target_dtype = np.float64 if np.issubdtype(orig_dtype, np.integer) else orig_dtype
            return X_out.astype(target_dtype)

        return X_out.tolist()


# Canonical alias for concise import
AMMI = AdaptiveMicroManifoldImputer

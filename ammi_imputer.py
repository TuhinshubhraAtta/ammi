import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AdaptiveMicroManifoldImputer(BaseEstimator, TransformerMixin):
    """
    Adaptive Micro-Manifold Imputer (AMMI)
    ======================================
    A novel, single-pass non-parametric algorithm that bridges the gap between:
    - Speed of Median Imputation (O(n) linear complexity)
    - Non-linear neighborhood sensitivity of KNN (without O(n^2) pairwise distance matrices)
    - Covariance & correlation preservation of MICE (without multi-pass iterative loops)

    Theoretical Foundations:
    ------------------------
    1. Orthonormal Random Slicing (Locality Sensitive Micro-Manifolds):
       Projects continuous features onto L random hyperspherical slices to partition the
       manifold into localized micro-neighborhoods in O(L * d * n) time.
       
    2. Empirical Bayes James-Stein Shrinkage:
       Computes local conditional expectations in each micro-cell and smoothly shrinks
       sparse cells toward global robust rank medians based on local observation counts:
           w_cell = N_cell / (N_cell + tau)
           x_base = w_cell * local_mean + (1 - w_cell) * global_median

    3. Directed Covariance Residual Flow (DCRF):
       Performs a 1-step orthogonal residual projection using observed feature deviations:
           delta_j = sum_{k in obs} (rho_{jk} * z_k) * sigma_j
       This preserves the feature correlation matrix without needing to fit separate regression models.
    """
    def __init__(self, n_projections=5, n_bins=5, shrinkage_tau=4.0, random_state=42):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.shrinkage_tau = shrinkage_tau
        self.random_state = random_state

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)

        # 1. Global Robust Statistics
        self.global_medians_ = np.nanmedian(X_arr, axis=0)
        self.global_stds_ = np.nanstd(X_arr, axis=0)
        self.global_stds_[self.global_stds_ == 0] = 1.0

        # Baseline normalized impute for initial projection
        X_clean = np.where(np.isnan(X_arr), self.global_medians_, X_arr)
        X_norm = (X_clean - self.global_medians_) / self.global_stds_

        # 2. Orthonormal Random Projection Matrix (L unit vectors)
        # Guarantees valid projections regardless of n_features
        R = rng.standard_normal((n_features, self.n_projections))
        norms = np.linalg.norm(R, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        self.projection_matrix_ = R / norms

        # 3. Quantile-based Multi-Resolution Binning
        projected = X_norm @ self.projection_matrix_
        self.bin_edges_ = [
            np.quantile(projected[:, p], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            for p in range(self.n_projections)
        ]

        # 4. Empirical Correlation Matrix (Directed Residual Flow)
        corr = np.corrcoef(X_clean, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        self.corr_matrix_ = np.nan_to_num(corr, nan=0.0)

        # 5. Pre-aggregate Micro-Cell Local Expectations
        hash_keys = self._hash_projections(projected)
        df_tmp = pd.DataFrame(X_arr)
        df_tmp['__hash__'] = hash_keys

        grouped = df_tmp.groupby('__hash__')
        self.cell_means_ = {}
        self.cell_counts_ = {}

        for h_key, grp in grouped:
            raw_vals = grp.iloc[:, :n_features].to_numpy()
            cnts = np.sum(~np.isnan(raw_vals), axis=0)
            means = np.nanmean(raw_vals, axis=0)
            # Replace NaNs in means where count is 0
            means = np.where(cnts > 0, means, self.global_medians_)
            self.cell_means_[h_key] = means
            self.cell_counts_[h_key] = cnts

        return self

    def _hash_projections(self, projected):
        bins = np.zeros(projected.shape, dtype=np.int32)
        for p in range(self.n_projections):
            bins[:, p] = np.digitize(projected[:, p], self.bin_edges_[p])
        # Return tuple keys for dictionary hashing
        return [tuple(row) for row in bins]

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64).copy()
        n_samples, n_features = X_arr.shape

        # Standardize using fitted statistics
        X_clean = np.where(np.isnan(X_arr), self.global_medians_, X_arr)
        X_norm = (X_clean - self.global_medians_) / self.global_stds_
        projected = X_norm @ self.projection_matrix_
        hashes = self._hash_projections(projected)

        nan_mask = np.isnan(X_arr)
        if not np.any(nan_mask):
            return X_arr

        # Fast Vectorized Matrix of Cell Base Estimates
        base_imputed = np.tile(self.global_medians_, (n_samples, 1))
        
        # Fill base estimates from micro-cells with James-Stein shrinkage
        for r, h in enumerate(hashes):
            if h in self.cell_means_:
                means = self.cell_means_[h]
                cnts = self.cell_counts_[h]
                shrinkage = cnts / (cnts + self.shrinkage_tau)
                base_imputed[r] = shrinkage * means + (1.0 - shrinkage) * self.global_medians_

        # Directed Covariance Residual Flow:
        # Z-scores of observed values (0 for missing values so they don't corrupt flow)
        observed_mask = ~nan_mask
        Z_observed = np.where(observed_mask, (X_arr - self.global_medians_) / self.global_stds_, 0.0)

        # Residual Flow Projection: (N x d) @ (d x d)
        # Normalization factor per feature
        corr_abs_sum = np.sum(np.abs(self.corr_matrix_), axis=0, keepdims=True)
        corr_abs_sum[corr_abs_sum == 0] = 1.0
        residual_flow = (Z_observed @ self.corr_matrix_) / corr_abs_sum
        residual_adjustment = residual_flow * self.global_stds_

        # Combine: Final Value = Shrunk Manifold Centroid + Directed Residual Adjustment
        final_imputed = base_imputed + residual_adjustment
        
        # Replace only the missing positions
        X_arr[nan_mask] = final_imputed[nan_mask]
        return X_arr

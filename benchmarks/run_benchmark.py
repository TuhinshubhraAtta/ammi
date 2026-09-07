"""
AMMI Benchmark Suite
====================
Reproducible comparative benchmark evaluating execution runtime, imputation error (RMSE),
and correlation distortion across common tabular imputation algorithms.

Usage:
    python benchmarks/run_benchmark.py [--n-samples 50000] [--missing-rate 0.15]
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.impute import SimpleImputer

# Optional IterativeImputer (MICE)
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    HAS_MICE = True
except ImportError:
    HAS_MICE = False

from ammi import AMMI


def generate_benchmark_data(n_samples: int = 50000, missing_rate: float = 0.15, seed: int = 42):
    """
    Generates synthetic tabular data with nonlinear relationships:
    x0 ~ Normal(10, 2)
    x1 ~ Normal(5, 1)
    x2 = 0.05 * x0^2 + 0.5 * x1 + Normal(0, 0.5)
    x3 = 1.2 * x0 - 0.8 * x1 + Normal(0, 0.5)
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(10.0, 2.0, size=n_samples)
    x1 = rng.normal(5.0, 1.0, size=n_samples)
    x2 = 0.05 * (x0 ** 2) + 0.5 * x1 + rng.normal(0.0, 0.5, size=n_samples)
    x3 = 1.2 * x0 - 0.8 * x1 + rng.normal(0.0, 0.5, size=n_samples)

    X_clean = np.column_stack([x0, x1, x2, x3])

    # Introduce MCAR missingness
    mask = rng.uniform(0.0, 1.0, size=X_clean.shape) < missing_rate
    X_missing = X_clean.copy()
    X_missing[mask] = np.nan

    return X_clean, X_missing, mask


def evaluate_imputer(name: str, imputer, X_clean: np.ndarray, X_missing: np.ndarray, mask: np.ndarray):
    """Measures fit_transform runtime, RMSE on missing values, and correlation distortion."""
    start_time = time.perf_counter()
    X_imputed = imputer.fit_transform(X_missing)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Ensure array format
    if hasattr(X_imputed, "to_numpy"):
        X_imputed = X_imputed.to_numpy()
    elif isinstance(X_imputed, list):
        X_imputed = np.array(X_imputed)

    # 1. RMSE on masked entries
    rmse = float(np.sqrt(np.mean((X_clean[mask] - X_imputed[mask]) ** 2)))

    # 2. Correlation Distortion (Mean absolute error of off-diagonal elements)
    n_features = X_clean.shape[1]
    off_diag = ~np.eye(n_features, dtype=bool)
    true_corr = np.corrcoef(X_clean, rowvar=False)
    imp_corr = np.corrcoef(X_imputed, rowvar=False)
    corr_distortion = float(np.mean(np.abs(true_corr[off_diag] - imp_corr[off_diag])))

    return {
        "name": name,
        "runtime_ms": elapsed_ms,
        "rmse": rmse,
        "corr_distortion": corr_distortion,
    }


def main():
    parser = argparse.ArgumentParser(description="Run AMMI comparative benchmark.")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of rows in benchmark dataset.")
    parser.add_argument("--missing-rate", type=float, default=0.15, help="Proportion of missing values per feature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"Generating benchmark dataset: {args.n_samples:,} rows x 4 features ({args.missing_rate:.0%} missing)...")
    X_clean, X_missing, mask = generate_benchmark_data(args.n_samples, args.missing_rate, args.seed)

    methods = [
        ("Global Mean", SimpleImputer(strategy="mean")),
        ("Global Median", SimpleImputer(strategy="median")),
    ]

    if HAS_MICE:
        methods.append((
            "MICE (BayesianRidge, 5 iter)",
            IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=args.seed)
        ))

    methods.append((
        "AMMI (Ours)",
        AMMI(n_projections=4, n_bins=4, shrinkage_tau=3.0, random_state=args.seed)
    ))

    results = []
    print("\nRunning benchmarks...")
    for name, imputer in methods:
        print(f"  Evaluating {name}...")
        res = evaluate_imputer(name, imputer, X_clean, X_missing, mask)
        results.append(res)

    print("\n" + "=" * 78)
    print(f"Benchmark Results ({args.n_samples:,} Rows x 4 Features, {args.missing_rate:.0%} Missing)")
    print("=" * 78)
    header = f"| {'Algorithm':<28} | {'Runtime (ms)':<14} | {'RMSE':<10} | {'Corr Distortion':<16} |"
    print(header)
    print("|" + "-" * 30 + "|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 18 + "|")
    for r in results:
        print(f"| {r['name']:<28} | {r['runtime_ms']:>12.1f} ms | {r['rmse']:>10.4f} | {r['corr_distortion']:>16.4f} |")
    print("=" * 78)


if __name__ == "__main__":
    main()

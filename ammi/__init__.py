"""
AMMI: Adaptive Micro-Manifold Imputer
====================================
A linear-time O(n) non-parametric imputation framework for missing tabular data.

Basic Usage:
------------
>>> from ammi import AMMI
>>> imputer = AMMI()
>>> X_clean = imputer.fit_transform(X_missing)
"""

from .imputer import AdaptiveMicroManifoldImputer, AMMI

__version__ = "1.0.0"
__author__ = "AMMI Authors"
__license__ = "MIT"
__all__ = ["AdaptiveMicroManifoldImputer", "AMMI", "__version__"]

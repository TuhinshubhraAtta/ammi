"""
AMMI: Adaptive Micro-Manifold Imputer
====================================

A fast, linear-time non-parametric missing value imputation framework
combining orthonormal random projection slicing, empirical Bayes shrinkage,
and correlation-weighted residual projection.
"""

from .imputer import AdaptiveMicroManifoldImputer, AMMI

__version__ = "1.0.0"
__author__ = "Tuhinshubhra Atta"
__license__ = "MIT"
__all__ = ["AdaptiveMicroManifoldImputer", "AMMI", "__version__"]

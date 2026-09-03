"""PhishGuard AI machine-learning package.

Contains the canonical URL feature-engineering implementation, dataset
loaders, the preprocessing pipeline and the model training entry point.

The backend imports :mod:`ml.features` directly so that training and
inference can never drift apart (train/serve skew).
"""

__version__ = "1.0.0"

import logging
from typing import Literal

import numpy as np
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    TomekLinks,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
)
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    KMeansSMOTE,
    SMOTENC,
    SMOTEN,
    ADASYN,
)

logger = logging.getLogger(__name__)

RESAMPLING_STRATEGIES = Literal[
    "random_under_sampler",
    "nearmiss",
    "tomeklinks",
    "allknn",
    "edited_nn",
    "repeated_edited_nn",
]


def balanced(y):
    """Check if the data is balanced."""
    class_counts = np.bincount(y)
    if class_counts[0] != class_counts[1]:
        return False
    return True


def inputs_size(X):
    """Return the size of the input data in Megabytes."""
    return X.nbytes // 1024


class Resampler:
    """
    A class for resampling data using different resampling strategies.

    Args:
    - strategy (str): The resampling strategy to use. Available strategies are:
        - "random_under_sampler": RandomUnderSampler
        - "nearmiss-1": NearMiss (version 1)
        - "nearmiss-2": NearMiss (version 2)
        - "nearmiss-3": NearMiss (version 3)
        - "tomeklinks": TomekLinks
        - "allknn": AllKNN
        - "edited_nn": EditedNearestNeighbours
        - "repeated_edited_nn": RepeatedEditedNearestNeighbours
        - "one_sided_selection": OneSidedSelection
        - "neighbourhood_cleaning_rule": NeighbourhoodCleaningRule
    - **kwargs: Additional keyword arguments to pass to the resampling method.

    Methods:
    - fit_resample(self, X, y): Fits the resampler to the data and returns the resampled data.

    """

    resampling_methods = {
        # Downsampling strategies
        "random_under_sampler": lambda **k: RandomUnderSampler(**k),
        "nearmiss": lambda **k: NearMiss(**k),
        "tomeklinks": lambda **k: TomekLinks(**k),
        "allknn": lambda **k: AllKNN(**k),
        "edited_nn": lambda **k: EditedNearestNeighbours(**k),
        "repeated_edited_nn": lambda **k: RepeatedEditedNearestNeighbours(**k),
        # Upsampling strategies
        "random_over_sampler": lambda **k: RandomOverSampler(**k),
        "smote": lambda **k: SMOTE(**k),
        "borderline_smote": lambda **k: BorderlineSMOTE(**k),
        "svm_smote": lambda **k: SVMSMOTE(**k),
        "kmeans_smote": lambda **k: KMeansSMOTE(**k),
        "smote_nc": lambda **k: SMOTENC(**k),
        "smote_n": lambda **k: SMOTEN(**k),
        "adasyn": lambda **k: ADASYN(**k),
    }

    def __init__(self, strategy: str = "random_under_sampler", **kwargs):

        if strategy not in self.resampling_methods:
            msg = (
                f"Invalid resampling strategy: {strategy}, "
                f"available strategies are: {list(self.resampling_methods.keys())}"
            )
            raise ValueError(msg)
        self._strategy = strategy
        self._resampler = self.resampling_methods[strategy](**kwargs)
        logger.info(f"Initialized resampler: {self._resampler.__class__.__name__}")

    def fit_resample(self, X, y, return_indices=False):
        """Fit the resampler to the data and return the resampled data."""
        self._validate_inputs_targets(X, y)
        X, shape = self._flatten_inputs(X)

        X_info = f"input size: {inputs_size(X)} MB, class counts: {np.bincount(y)}"
        logger.info("Resampling data with %s", X_info)
        X, y = self._resampler.fit_resample(X, y)
        indices = self._resampler.sample_indices_
        X_info = f"input size: {inputs_size(X)} MB, class counts: {np.bincount(y)}"
        logger.info("Resampling complete, %s", X_info)

        if self._majority_became_minority(y):
            logger.warning("Majority class became minority after resampling")

        if not balanced(y):
            logger.debug("Requiring further resampling, applying RandomUnderSampler")
            rus = RandomUnderSampler(random_state=0)
            X, y = rus.fit_resample(X, y)
            indices = indices[self._resampler.sample_indices_]
            X_info = f"input size: {inputs_size(X)} MB, class counts: {np.bincount(y)}"
            logger.info("Resampling complete, %s", X_info)

        if return_indices:
            return indices
        X = self._restore_inputs_shape(X, shape)
        return X, y

    __call__ = fit_resample

    @staticmethod
    def _validate_inputs_targets(X, y):
        """Validate the input data and targets."""
        if not isinstance(X, np.ndarray):
            msg = "Input data must be a numpy array"
            raise ValueError(msg)
        if not isinstance(y, np.ndarray):
            msg = "Target labels must be a numpy array"
            raise ValueError(msg)
        if len(np.unique(y)) != 2:
            msg = "Invalid number of classes"
            raise ValueError(msg)

    @staticmethod
    def _flatten_inputs(X):
        """Flatten the input data."""
        if X.ndim > 2:
            shape = X.shape[1:]
            return X.reshape([X.shape[0], -1]), shape
        return X, None

    @staticmethod
    def _restore_inputs_shape(X, shape):
        """Restore the original shape of the input data."""
        if shape is not None:
            X = X.reshape([X.shape[0], *shape])
        return X

    @staticmethod
    def _majority_became_minority(y):
        """Check if the majority class becomes the minority class after resampling."""
        class_counts = np.bincount(y)
        if class_counts[0] < class_counts[1]:
            return True
        return False

    def __repr__(self):
        return f"Resampler(strategy={self._strategy})"

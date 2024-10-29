"""Collection of tools to log performance of a model."""

from typing import Any

import numpy as np


def accuracy_score(y_true, y_pred):
    """Return the % of good prediction a model done."""
    return np.count_nonzero(y_true == y_pred) / len(y_pred) * 100


def binary_cross_entropy(y_true, y_pred):
    """Implementation of binary cross entropy algorithm.

    binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]
    """
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """Implementation of categorical cross entropy algorithm.

    categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """
    return -np.sum(y * np.log(p))

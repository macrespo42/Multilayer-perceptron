"""Collection of algorithm used by machine learning models."""

from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid on a numpy array.

    sigmoid(x: np.ndarray) -> np.ndarray:
    """
    return 1 / (1 + np.exp(x))


def softmax(z: np.ndarray) -> np.ndarray:
    """Implementation of the softmax algorithm.

    softmax(z: np.ndarray) -> np.ndarray:
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def rlu(z: np.ndarray) -> np.ndarray:
    """Implementation of the rectified linear unit algorithm.

    rlu(z: np.ndarray) -> np.ndarray
    """
    z[z < 0] = 0
    return z


def binary_cross_entropy(y_true, y_pred):
    """Implementation of binary cross entropy algorithm.

    binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]
    """
    np.clip(y_pred, 1e-7, 1 - 1e-7)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """Implementation of categorical cross entropy algorithm.

    categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """
    return -np.sum(y * np.log(p))

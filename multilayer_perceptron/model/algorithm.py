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


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """Implementation of binary cross entropy algorithm.

    binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]
    """
    return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p), axis=1)


def categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """Implementation of categorical cross entropy algorithm.

    categorical_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.floating[Any]:
    """
    return -np.sum(y * np.log(p))

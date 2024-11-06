"""Collection of algorithm used by machine learning models."""

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
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def rlu(z: np.ndarray) -> np.ndarray:
    """Implementation of the rectified linear unit algorithm.

    rlu(z: np.ndarray) -> np.ndarray
    """
    return np.maximum(0, z)

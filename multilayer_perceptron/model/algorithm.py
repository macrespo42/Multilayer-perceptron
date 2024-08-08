"""Collection of algorithm used in machine learning models."""

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
    return exp_z / np.sum(exp_z)


def rlu(z: np.ndarray) -> np.ndarray:
    """Implementation of the rectified linear unit algorithm.

    rlu(z: np.ndarray) -> np.ndarray
    """
    z[z < 0] = 0
    return z

"""Collection of algorithm used in machine learning models."""

import numpy as np


def sigmoid(x) -> np.float64:
    """Apply sigmoid on a numpy array.

    sigmoid(x) -> np.float64
    """
    return 1 / (1 + np.exp(x))


def softmax(z, k, j) -> np.float64:
    """Implementation of the softmax algorithm.

    def softmax(z):
    """
    return np.exp(z) / np.sum(np.exp(z[j:k]))

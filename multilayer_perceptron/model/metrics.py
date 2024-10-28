"""Collection of tools to log performance of a model."""

import numpy as np

def accuracy_score(y_true, y_pred):
    """Return the % of good prediction a model done."""
    return np.count_nonzero(y_true == y_pred) / len(y_pred) * 100

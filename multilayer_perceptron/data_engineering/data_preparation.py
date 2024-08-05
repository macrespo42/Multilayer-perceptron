"""Prepare data before giving it to a model."""

import pandas as pd
import numpy as np


def encode_categorical_variables(column: pd.Series, a: str, b: str) -> pd.Series:
    """Encode categorical variables numerics equivalent 1 and 0.

    encode_categorical_variables(column: pd.Series, a: str, b: str) -> pd.Series
    """
    return column.map({a: 1, b: 0})


def standardization(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """Standardize given column of a dataset using z-score.

    standardization(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """
    for column in columns:
        x = df[column]
        df[column] = (x - x.mean()) / x.std()
    return df


def normalization(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """Normalize given column of a dataframe using min-max.

    normalization(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """
    for column in columns:
        x = df[column]
        df[column] = (x - x.min()) / (x - x.max())
    return df

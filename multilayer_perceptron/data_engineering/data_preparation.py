"""Prepare data before giving it to a model."""

import numpy as np
import pandas as pd


def encode_categorical_variables(column, a: str, b: str) -> np.ndarray:
    """Encode categorical variables numerics equivalent 1 and 0.

    encode_categorical_variables(column: pd.Series, a: str, b: str) -> pd.Series
    """
    return np.vstack(column.map({a: [1, 0], b: [0, 1]}).values)


def standardize(df, columns) -> pd.DataFrame:
    """Standardize given column of a dataset using z-score.

    standardize(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """
    for column in columns:
        x = df[column]
        df.loc[:, column] = (x - x.mean()) / x.std()
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize given column of a dataframe using min-max.

    normalize(df: pd.DataFrame, columns: np.ndarray) -> pd.DataFrame:
    """
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

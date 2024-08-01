"""Play module."""

from numpy import float64
import pandas as pd
import sys


def add_WDBC_headers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add Wisconsin Diagnostic Breast Cancer attributes as column names of the given dataframe.

    add_WDBC_headers(dataframe: pd.DataFrame) -> pd.DataFrame
    """
    COLUMN_NAMES = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    dataframe.columns = COLUMN_NAMES
    return dataframe


if __name__ == "__main__":
    COLUMN_NAMES = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    df = pd.read_csv(sys.argv[1])
    first_line = df.columns.astype(float64)
    print(first_line)
    print(f"first line: {type(df.iloc[0])}")
    print(df.head())

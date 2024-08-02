"""Parsing dataset module."""

import pandas as pd
import sys


def load(path: str, names=None) -> pd.DataFrame:
    """Open a dataset and return it.

    load(path: str) -> Dataset
    """
    data_file = None
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("path isn't a csv file")
        if names:
            data_file = pd.read_csv(path, names=names)
        else:
            data_file = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found, try to run the script from the root of multilayer_perceptron")
    except PermissionError:
        raise PermissionError(f"You don't have permission to read ${path}")
    return data_file


def read_csv_with_WDBC_headers(csv_path: str) -> pd.DataFrame:
    """Read csv and Add Wisconsin Diagnostic Breast Cancer attributes as column names.

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
    dataframe = load(csv_path, names=COLUMN_NAMES)
    return dataframe


if __name__ == "__main__":
    df = read_csv_with_WDBC_headers(sys.argv[1])
    print(df.head())
    print(df.loc[df.diagnosis == "B"].head())

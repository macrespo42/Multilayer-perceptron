"""Separate program."""

import math
import sys

from data_engineering.parse import read_csv_with_WDBC_headers


def separate(path: str, name="data", train_percentage=70) -> None:
    """Separate given csv into 2 file: data_train (first 70% of data) and data_test (last 30%).

    separate(path: str, name="data") -> None
    """
    if train_percentage >= 99:
        raise ValueError("The train percentage is to high")
    df = read_csv_with_WDBC_headers(path)
    # df = df.sample(frac=1)
    train_stop_idx = math.floor((train_percentage / 100) * df.shape[0])

    data_train = df.iloc[0:train_stop_idx]
    data_test = df.iloc[train_stop_idx::]

    train_path = f"datasets/{name}_train.csv"
    test_path = f"datasets/{name}_test.csv"

    try:
        data_train.to_csv(train_path)
        data_test.to_csv(test_path)
    except Exception:
        print("Cannot save split data. Try running script from the root directory")
        sys.exit(1)

    print(f"training data saved at: {train_path}")
    print(f"test data saved at: {test_path}")


def main() -> None:
    """Main function in charge of executing separate and handle errors.

    main() -> None
    """
    if not len(sys.argv) == 2:
        raise ValueError("Please provide the dataset to separate as argument of this script")
    separate(sys.argv[1])


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}")

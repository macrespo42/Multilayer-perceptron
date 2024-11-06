"""Training program for the mlp model."""

import argparse

import pandas as pd
from data_engineering import data_preparation
from model import mlp


def list_of_ints(arg):
    """Parse a list of ints from stdin."""
    return list(map(int, arg.split(",")))


def train():
    """Placeholder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_train", help="path to the training dataset", type=str)
    parser.add_argument("data_test", help="path to the test dataset", type=str)
    parser.add_argument("--epochs", help="number of iteration the model as to train", type=int)
    parser.add_argument("--learning_rate", help="which speed the model as to learn", type=float)
    parser.add_argument("--layer", help="Number of neurons for each hidden layers", type=list_of_ints)
    args = parser.parse_args()

    breast_cancer_data_train = None
    breast_cancer_data_test = None

    try:
        breast_cancer_data_train = pd.read_csv(args.data_train)
        breast_cancer_data_test = pd.read_csv(args.data_test)
    except Exception:
        print("Can't find dataset")
        exit(1)

    feature_names = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
    ]

    X_train, y_train = (breast_cancer_data_train[feature_names], breast_cancer_data_train["diagnosis"])
    X_test, y_test = (breast_cancer_data_test[feature_names], breast_cancer_data_test["diagnosis"])

    X_norm = data_preparation.normalize(X_train)
    y_norm = data_preparation.encode_categorical_variables(y_train, "B", "M")

    X_test = data_preparation.normalize(X_test)
    y_test = data_preparation.encode_categorical_variables(y_test, "B", "M")

    input_shape = len(X_norm.columns)
    output_shape = 2
    network = []

    if args.layer is not None:
        network.append(mlp.DenseLayer(input_shape, input_shape, activation="rlu"))
        prev_shape = input_shape
        for i in range(len(args.layer)):
            network.append(mlp.DenseLayer(prev_shape, args.layer[i], activation="rlu"))
            prev_shape = args.layer[i]
        network.append(
            mlp.DenseLayer(prev_shape, output_shape, activation="softmax"),
        )
    else:
        network = [
            mlp.DenseLayer(input_shape, input_shape, activation="rlu"),
            mlp.DenseLayer(24, 32, activation="rlu"),
            mlp.DenseLayer(32, 32, activation="rlu"),
            mlp.DenseLayer(32, 32, activation="rlu"),
            mlp.DenseLayer(32, output_shape, activation="softmax"),
        ]

    epochs = 60_000 if not args.epochs else args.epochs
    learning_rate = 0.01 if not args.learning_rate else args.learning_rate

    model = mlp.MultilayerPerceptron(network, epochs=epochs, learning_rate=learning_rate)
    model.fit(X_norm, y_norm, X_test, y_test)
    model.save()


if __name__ == "__main__":
    train()

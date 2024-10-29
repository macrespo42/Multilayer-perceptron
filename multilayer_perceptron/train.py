"""Training program for the mlp model."""

import pandas as pd
from data_engineering import data_preparation
from model import mlp
from model.metrics import accuracy_score, binary_cross_entropy


def train():
    """Placeholder."""
    breast_cancer_data_train = pd.read_csv("datasets/data_train.csv")
    breast_cancer_data_test = pd.read_csv("datasets/data_test.csv")

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

    network = [
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 2, activation="softmax"),
    ]

    model = mlp.MultilayerPerceptron(X_norm, y_norm, network, epochs=7_000, learning_rate=0.1)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)
    print(f"ACCURRACY: { accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)) }")
    print(f"MY LOSS: {binary_cross_entropy(y_test.argmax(axis=1), y_pred[:, 1])}")


if __name__ == "__main__":
    train()

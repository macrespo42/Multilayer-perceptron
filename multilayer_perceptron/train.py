"""Training program for the mlp model."""

import numpy as np
from data_engineering import data_preparation, parse
from model import mlp
from model.metrics import accuracy_score
from sklearn.metrics import log_loss
from model.algorithm import binary_cross_entropy

def train():
    """Placeholder."""
    breast_cancer_data = parse.read_csv_with_WDBC_headers("datasets/data.csv")

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

    X, y = (breast_cancer_data[feature_names], breast_cancer_data["diagnosis"])

    X_norm = data_preparation.normalize(X, X.columns)
    y_norm = data_preparation.encode_categorical_variables(y, "B", "M")

    network = [
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 24, activation="rlu"),
        mlp.DenseLayer(24, 2, activation="softmax"),
    ]

    model = mlp.MultilayerPerceptron(X_norm, y_norm, network, epochs=10, learning_rate=0.1)
    model.fit()
    y_pred = model.predict(X_norm).argmax(axis=1)
    print(f"ACCURRACY: { accuracy_score(y_norm.argmax(axis=1), y_pred) }")
    print(f"LOSS: {log_loss(y_norm.argmax(axis=1), y_pred)}")
    # print(f"MY LOSS: {binary_cross_entropy(y_norm.argmax(axis=1), y_pred)}")

if __name__ == "__main__":
    train()

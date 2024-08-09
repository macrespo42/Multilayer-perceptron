"""Training program for the mlp model."""

import numpy as np
from data_engineering import parse
from data_engineering import data_preparation
from model import mlp

import nnfs
from nnfs.datasets import spiral_data


def model_42():
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

    X_norm = data_preparation.standardize(X, X.columns)
    y_norm = data_preparation.encode_categorical_variables(y, "B", "M")

    network = [
        mlp.DenseLayer(24, 3, activation="rlu"),
        mlp.DenseLayer(3, 3, activation="rlu"),
        mlp.DenseLayer(3, 3, activation="rlu"),
        mlp.DenseLayer(3, 2, activation="sigmoid"),
    ]

    model = mlp.MultilayerPerceptron(network)
    model.forward(X_norm)
    print(model.output)
    print(y_norm.values)


class Loss:
    """Placeholder."""

    def calculate(self, output, y):
        """Placeholder."""
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)

        return data_loss

    def forward(self, y_pred, y_true):
        """Placeholder."""
        print(y_true + y_pred)
        return np.array([1, 2, 3, 4])


class Loss_CategoricalCrossentropy(Loss):
    """Placeholder."""

    def forward(self, y_pred, y_true):
        """Placeholder."""
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.log(correct_confidences)


if __name__ == "__main__":
    # Multiclassifier scenario
    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    dense1 = mlp.DenseLayer(2, 3, activation="rlu")
    dense2 = mlp.DenseLayer(3, 3, activation="softmax")

    loss_function = Loss_CategoricalCrossentropy()

    model = mlp.MultilayerPerceptron([dense1, dense2])
    model.forward(X)
    loss = loss_function.calculate(model.output, y)

    predictions = np.argmax(model.output, axis=1)
    accuracy = np.mean(predictions == y)
    print(model.output[:5])
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

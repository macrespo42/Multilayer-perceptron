"""Training program for the mlp model."""

import numpy as np
from data_engineering import data_preparation, parse
from model import algorithm, mlp


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
        mlp.DenseLayer(24, 10, activation="rlu"),
        mlp.DenseLayer(10, 10, activation="rlu"),
        mlp.DenseLayer(10, 10, activation="rlu"),
        mlp.DenseLayer(10, 2, activation="softmax"),
    ]

    model = mlp.MultilayerPerceptron(network)
    model.forward(X_norm)
    print(model.output)
    output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
    target = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    print(algorithm.categorical_cross_entropy(target, output))


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
    model_42()

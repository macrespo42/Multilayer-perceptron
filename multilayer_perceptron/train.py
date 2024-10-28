"""Training program for the mlp model."""

import numpy as np
from data_engineering import data_preparation, parse
from model import mlp
from sklearn.metrics import accuracy_score


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

    X_norm = data_preparation.normalize(X, X.columns)
    y_norm = data_preparation.encode_categorical_variables(y, "B", "M")

    network = [
        mlp.DenseLayer(24, 16, activation="rlu"),
        mlp.DenseLayer(16, 32, activation="rlu"),
        mlp.DenseLayer(32, 16, activation="rlu"),
        mlp.DenseLayer(16, 2, activation="softmax"),
    ]

    model = mlp.MultilayerPerceptron(X_norm, y_norm, network, epochs=50_000)
    model.fit()
    y_pred = model.predict(X_norm).argmax(axis=1)
    print(f"ACCURRACY: { accuracy_score(y_norm.argmax(axis=1), y_pred) }")


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
    model_42()

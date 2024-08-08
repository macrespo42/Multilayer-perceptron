"""Multilayer perceptron model."""

import numpy as np
from algorithm import rlu, sigmoid, softmax

# For testing purpose to remove
from nnfs.datasets import spiral_data


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor."""
        self.n_neurons = n_neurons

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, self.n_neurons))
        self.activation = activation

    def forward(self, inputs) -> None:
        """Output of the perceptron.

        forward(self, inputs) -> Any
        """
        self.output = np.dot(inputs, self.weights) + self.bias

    def activate(self) -> np.ndarray | None:
        """Apply activation function to the neurons output.

        activate(self) -> np.ndarray | None
        """
        activate_output = None
        if self.output is not None:
            if self.activation == "sigmoid":
                activate_output = sigmoid(self.output)
            elif self.activation == "softmax":
                activate_output = softmax(self.output)
            elif self.activation == "rlu":
                activate_output = rlu(self.output)
            else:
                raise NotImplementedError(f"{self.activation} activation function is not implemented.")
        return activate_output


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    pass


if __name__ == "__main__":
    np.random.seed(0)
    X = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    X, y = spiral_data(100, 3)

    layer1 = DenseLayer(2, 3, activation="rlu")
    layer2 = DenseLayer(3, 3, activation="softmax")

    layer1.forward(X)
    layer2.forward(layer1.activate())

    output = layer2.activate()
    if output is not None:
        print(output[:5])
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy}")

    # softmax_output = np.array([0.7, 0.1, 0.2])
    # target_output = np.array([1, 0, 0])

    # print(f"binary_cross_entropy: {binary_cross_entropy(target_output, softmax_output)}")
    # print(f"categorical_cross_entropy: {categorical_cross_entropy(target_output, softmax_output)}")

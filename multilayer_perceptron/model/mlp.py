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

        ACTIVATION = {"sigmoid": sigmoid, "softmax": softmax, "rlu": rlu}

        self.activation = ACTIVATION.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"{self.activation} activation function is not implemented.")

    def forward(self, inputs) -> None:
        """Output of the perceptron.

        forward(self, inputs) -> Any
        """
        self.output = np.dot(inputs, self.weights) + self.bias

    def activate(self):
        """Apply activation function to the neurons output.

        activate(self)
        """
        if self.output is not None:
            self.output = self.activation(self.output)
        else:
            print("WARNING: Forward layer before activate it")


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    def __init__(self, network) -> None:
        """MLP constructor."""
        self.network = network

    def forward(self, inputs):
        """FeedForward in mlp."""
        for layer in self.network:
            layer.forward(inputs)
            layer.activate()
            inputs = layer.output


if __name__ == "__main__":
    np.random.seed(0)
    X = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    X, y = spiral_data(100, 3)

    layer1 = DenseLayer(2, 3, activation="rlu")
    layer2 = DenseLayer(3, 3, activation="softmax")

    layer1.forward(X)
    layer2.forward(layer1.activate())

    # softmax_output = np.array([0.7, 0.1, 0.2])
    # target_output = np.array([1, 0, 0])

    # print(f"binary_cross_entropy: {binary_cross_entropy(target_output, softmax_output)}")
    # print(f"categorical_cross_entropy: {categorical_cross_entropy(target_output, softmax_output)}")

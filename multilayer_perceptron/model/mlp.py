"""Multilayer perceptron model."""

import numpy as np
from algorithm import sigmoid, softmax


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor."""
        self.n_neurons = n_neurons

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, self.n_neurons))

        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "softmax":
            self.activation = softmax
        else:
            raise NotImplementedError(f"{activation} activation function is not implemented.")

    def forward(self, inputs):
        """Output of the perceptron."""
        self.output = np.dot(inputs, self.weights) + self.bias


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    pass


if __name__ == "__main__":
    np.random.seed(0)
    X = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    layer1 = DenseLayer(4, 5)
    layer2 = DenseLayer(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)

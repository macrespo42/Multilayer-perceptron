"""Multilayer perceptron model."""

import numpy as np
from algorithm import sigmoid, softmax

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

    def forward(self, inputs):
        """Output of the perceptron."""
        self.output = np.dot(inputs, self.weights) + self.bias

    def activate(self):
        """Placeholder."""
        activate_output = None
        if self.output:
            if self.activation == "sigmoid":
                activate_output = sigmoid(self.output)
            elif self.activation == "softmax":
                activate_output = softmax(self.output)
            else:
                raise NotImplementedError(f"{self.activation} activation function is not implemented.")
        return activate_output


class ActivationReLU:
    """Placeholder."""

    def forward(self, inputs):
        """Placeholder."""
        self.output = np.maximum(0, inputs)


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    pass


if __name__ == "__main__":
    np.random.seed(0)
    X = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    X, y = spiral_data(100, 3)

    layer1 = DenseLayer(2, 5)
    activation1 = ActivationReLU()

    layer1.forward(X)
    activation1.forward(layer1.output)
    print(activation1.output)

"""Multilayer perceptron model."""

import numpy as np
from algorithm import sigmoid, softmax


class Layer:
    """Layer of a mlp."""

    def __init__(self, n_perceptron, activation="sigmoid") -> None:
        """Layer constructor."""
        # TODO only for testing
        self.inputs = [1, 2, 3, 2.5]
        # END OF TODO
        self.weights = np.array([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
        self.bias = [2, 3, 0.5]
        self.n_perceptron = n_perceptron

        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "softmax":
            self.activation = softmax

    def output(self):
        """Placeholder."""
        outputs = []
        for i in range(len(self.weights)):
            outputs.append(np.dot(self.inputs, self.weights[i]) + self.bias[i])
        return outputs


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    pass


if __name__ == "__main__":
    layer = Layer(4, activation="sigmoid")
    print(layer.output())

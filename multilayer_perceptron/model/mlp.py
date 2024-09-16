"""Multilayer perceptron model."""

import numpy as np

from . import algorithm


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor."""
        self.n_neurons = n_neurons

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, self.n_neurons))

        ACTIVATION = {"sigmoid": algorithm.sigmoid, "softmax": algorithm.softmax, "rlu": algorithm.rlu}

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

    def __init__(self, network: list[DenseLayer]) -> None:
        """MLP constructor."""
        self.network = network

    def forward(self, inputs):
        """FeedForward in mlp."""
        if len(self.network) <= 0:
            return None
        for layer in self.network:
            layer.forward(inputs)
            layer.activate()
            inputs = layer.output
        output_layer = self.network[-1]
        self.output = output_layer.output

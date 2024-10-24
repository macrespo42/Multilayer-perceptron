"""Multilayer perceptron model."""

from os import walk

import numpy as np

from . import algorithm


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor."""
        self.n_neurons = n_neurons

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, self.n_neurons))
        self.dw = None
        self.db = None

        ACTIVATION = {"sigmoid": algorithm.sigmoid, "softmax": algorithm.softmax, "rlu": algorithm.rlu}

        self.activation = ACTIVATION.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"{self.activation} activation function is not implemented.")

    def activate(self):
        """Apply activation function to the neurons output.

        activate(self)
        """
        if self.output is not None:
            self.output = self.activation(self.output)
        else:
            print("WARNING: Forward layer before activate it")

    def forward_propagation(self, inputs) -> None:
        """Output of the perceptron.

        forward(self, inputs) -> Any
        """
        self.output = np.dot(inputs, self.weights) + self.bias
        self.activate()


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    def __init__(self, X: np.ndarray, y: np.ndarray, network: list[DenseLayer], learning_rate=0.1, epochs=1000) -> None:
        """MLP constructor."""
        self.X = X
        self.y = y
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self) -> None:
        """FeedForward in mlp.

        forward(self, inputs: np.ndarray) -> None
        """
        if len(self.network) <= 0:
            return None
        inputs = self.X
        for layer in self.network:
            layer.forward_propagation(inputs)
            inputs = layer.output
        output_layer = self.network[-1]
        self.output = output_layer.output

    def backward(self) -> None:
        """Backward propagation through network layers.

        backward(self, inputs: np.ndarray) -> None
        """
        m = len(self.y)
        C = len(self.network) // 2
        print(f"C: {C} | OR: {len(self.network)}")
        breakpoint()

        dZ = self.network[C].output - self.y
        for c in reversed(range(1, C + 1)):
            self.network[c].dw = 1 / m * np.dot(dZ, self.network[c].output.T)
            self.network[c].db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.network[c].weights.T, dZ) * self.network[c - 1].output * (1 - self.network[c - 1])

    def update(self) -> None:
        """Update weights and bias for each layers.

        update(self) -> None
        """
        C = len(self.network) // 2
        for c in range(1, C + 1):
            pass

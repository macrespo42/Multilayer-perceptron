"""Multilayer perceptron model."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from . import algorithm, metrics


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor."""
        np.random.seed(1)
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
        np.random.seed(1)
        self.X = X
        self.y = y
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output = None

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
        dZ = self.network[-1].output - self.y
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            prev_layer = self.network[i - 1] if i > 0 else self.X

            prev_output = prev_layer.output if i > 0 else prev_layer

            layer.dw = 1 / m * np.dot(prev_output.T, dZ)
            layer.db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dZ = np.dot(dZ, layer.weights.T) * prev_output * (1 - prev_output)

    def update(self) -> None:
        """Update weights and bias for each layers.

        update(self) -> None
        """
        for layer in self.network:
            layer.weights = layer.weights - self.learning_rate * layer.dw
            layer.bias = layer.bias - self.learning_rate * layer.db

    def fit(self) -> None:
        """Fit the model with givens X and y.

        fit(self) -> None
        """
        accuracies = {"train": [], "test": []}
        losses = {"train": [], "test": []}
        for _ in tqdm(range(self.epochs)):
            self.forward()
            accuracies["train"].append(
                metrics.accuracy_score(self.y.argmax(axis=1), self.network[-1].output.argmax(axis=1))
            )
            losses["train"].append(metrics.binary_cross_entropy(self.y.argmax(axis=1), self.network[-1].output[:, 1]))
            self.backward()
            self.update()

        print(np.min(losses["train"]))
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses["train"], label="training loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(accuracies["train"], label="training acc")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output."""
        self.forward()
        return np.array(self.network[-1].output)

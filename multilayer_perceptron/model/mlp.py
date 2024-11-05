"""Multilayer perceptron model."""

import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from . import algorithm, metrics

SEED = 3


class DenseLayer:
    """Layer of a mlp."""

    def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None:
        """Layer constructor.

        def __init__(self, n_inputs, n_neurons, activation="sigmoid") -> None
        """
        np.random.seed(SEED)
        self.n_neurons = n_neurons

        limit = sqrt(6.0 / n_inputs)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(n_inputs, n_neurons))
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
            return self.activation(self.output)
        else:
            print("WARNING: Forward layer before activate it")

    def forward_propagation(self, inputs) -> Any:
        """Output of the perceptron.

        forward(self, inputs) -> Any
        """
        self.output = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate()
        return self.output


class MultilayerPerceptron:
    """Multilayer perceptron model."""

    def __init__(self, network: list[DenseLayer], learning_rate=0.1, epochs=1000) -> None:
        """MLP constructor.
        
        __init__(self, network: list[DenseLayer], learning_rate=0.1, epochs=1000) -> None
        """
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output = None

    def forward(self, X) -> None:
        """FeedForward in mlp.

        forward(self, inputs: np.ndarray) -> None
        """
        if len(self.network) <= 0:
            return None
        inputs = X
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
            layer.weights -=   self.learning_rate * layer.dw
            layer.bias -=  self.learning_rate * layer.db

    def fit(self, X: np.ndarray, y: np.ndarray, X_test, y_test) -> None:
        """Fit the model with givens X and y.

        fit(self) -> None
        """
        self.X = X
        self.y = y
        accuracies = {"train": [], "test": []}
        losses = {"train": [], "test": []}
        for i in range(self.epochs):
            self.forward(self.X)
            accuracies["train"].append(
                metrics.accuracy_score(self.y.argmax(axis=1), self.network[-1].output.argmax(axis=1))
            )
            loss = metrics.binary_cross_entropy(self.y.argmax(axis=1), self.network[-1].output[:, 1])
            losses["train"].append(loss)
            self.backward()
            self.update()

            y_pred = self.predict(X_test)
            accuracies["test"].append(metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
            train_loss = metrics.binary_cross_entropy(y_test.argmax(axis=1), y_pred[:, 1])
            losses["test"].append(train_loss)
            print(f"Epochs {i}/{self.epochs} - loss: {loss} - val_loss: {train_loss}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses["train"], label="training loss")
        plt.plot(losses["test"], label="test loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(accuracies["train"], label="train acc")
        plt.plot(accuracies["test"], label="test acc")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output."""
        if len(self.network) <= 0:
            return None
        inputs = X
        for layer in self.network:
            inputs = layer.forward_propagation(inputs)
        return np.array(inputs)

    def save(self, path="model.npy") -> None:
        """Save current model as npy file."""
        if len(self.network) <= 0:
            return
        with open(path, "wb") as model:
            pickle.dump(self, model)
        print(f"model saved at {path}")

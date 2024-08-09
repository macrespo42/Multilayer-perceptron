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

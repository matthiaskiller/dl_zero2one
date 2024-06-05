
import os
import pickle
import numpy as np

from dl_zero2one.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super(Classifier, self).__init__("classifier")

        self.num_features = num_features
        self.W = None

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        Args:
            weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        Args:
            X: N x D array of training data. Each row is a D-dimensional point.
        Returns:
            Predicted labels for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        # save the samples for the backward pass
        self.cache = X
        # output variable
        y = self.sigmoid(np.dot(X, self.W))

        return y

    def backward(self, y):
        """
        Performs the backward pass of the model.

        Args:
            y: N x 1 array. The output of the forward pass.
        Returns:
            Gradient of the model output (y=sigma(X*W)) wrt W
        """
        assert self.cache is not None, "run a forward pass before the backward pass"

        dW = y * (1 - y) * self.cache

        return dW

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function

        Args:
            x: input of the sigmoid, np.array of any shape
        Returns:
            output of the sigmoid with same shape as input vector x
        """

        out = 1 / (1 + np.exp(-x))

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))

# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: Riya Venugopal Shetty -- rishett
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._y_classes = None
        self._h_weights = None
        self._h_bias = None
        self._h_output = None
        self._o_weights = None
        self._o_bias = None
        self._o_output = None
        self._output_nodes = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.
        Returns:
            None.
        """
        self._X = X
        self._input_nodes = self._X.shape[1]
        self._y = one_hot_encoding(y)
        self._y_classes = np.unique(y)
        self._output_nodes = self._y.shape[1]

        np.random.seed(42)
        self._h_weights = np.random.randn(self._input_nodes, self.n_hidden)
        self._h_bias = np.random.randn(self.n_hidden)
        self._h_bias = self._h_bias[:, np.newaxis] # column matrix
        self._o_weights = np.random.randn(self.n_hidden, self._output_nodes)
        self._o_bias = np.random.randn(self._output_nodes)
        self._o_bias = self._o_bias[:, np.newaxis] # column matrix


    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)

        # Get the new outputs from the network
        for e in range(self.n_iterations):
            CE_error = np.array([])
            p = np.random.permutation(len(self._X))
            self._X = self._X[p]
            self._y = self._y[p]
            for i in range(len(self._X)):
                X_i = self._X[i]
                y_i = self._y[i]

                # Feedforward step

                # Layer 1
                z_h = self._h_weights.T.dot(X_i[:, np.newaxis])
                z_h = z_h + self._h_bias
                self._h_output = self.hidden_activation(z_h)  # _h_output size m * 1
                # Layer 2
                z_o = self._o_weights.T.dot(self._h_output)
                z_o = z_o + self._o_bias
                self._o_output = self._output_activation(np.array([z_o.flatten()])) # 1 * output_nodes
                # Get error for the point
                CE_error = np.append(CE_error,cross_entropy(y_i[:, np.newaxis], self._o_output.T))

                # Backpropagation

                # Layer 2 weights and biases update
                err = self._o_output - np.array([y_i]) # 1 * output_nodes
                dCE_dW2 = np.dot(self._h_output , err) # no_hidden * no_output
                dCE_dW2 = self.learning_rate * dCE_dW2
                self._o_weights -= dCE_dW2 # no_hidden * no_output
                self._o_bias -= self.learning_rate * err.T

                # Layer 1 weights and biases update
                err = np.dot(err,self._o_weights.T) # 1 * no_hidden
                dactive = self.hidden_activation(z_h, derivative = True) # no_hidden * 1
                err = err * dactive.T # 1 * no_hidden
                dCE_dW2 = np.dot(X_i[:,np.newaxis],err) # no_inputs * no_hidden
                dCE_dW2 *= self.learning_rate
                self._h_weights -= dCE_dW2
                self._h_bias -= self.learning_rate * err.T
            if e%20 == 0:
                self._loss_history.append(np.sum(CE_error))





    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        outputs = np.array([])
        for i in range(len(X)):
            X_i = X[i]
            # Feedforward step

            # Layer 1
            z_h = self._h_weights.T.dot(X_i[:, np.newaxis])
            z_h = z_h + self._h_bias
            h_output = self.hidden_activation(z_h)  # _h_output size m * 1
            # Layer 2
            z_o = self._o_weights.T.dot(h_output)
            z_o = z_o + self._o_bias
            output = self._output_activation(np.array([z_o.flatten()])) # 1 * output_nodes
            pred = np.argmax(output.flatten())
            outputs = np.append(outputs, self._y_classes[pred])
        return outputs


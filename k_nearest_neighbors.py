# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Riya Venugopal Shetty -- rishett
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
import heapq
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """


        n = len(self._X)
        y_pred = []
        for xi in X:
            neighbours = []
            for i in range(n):
                point = self._X[i]
                yi = self._y[i]
                dist = self._distance(point, xi)
                if len(neighbours) < self.n_neighbors:
                    # Using min heap to store the K neighbours
                    # Each Element is stored as (negative of distance, yi of the corresponding point)
                    heapq.heappush(neighbours, (-dist, yi))
                else:
                    most_distant_point = heapq.heappop(neighbours)
                    if dist < (-most_distant_point[0]):
                        heapq.heappush(neighbours, (-dist, yi))
                    else:
                        heapq.heappush(neighbours, most_distant_point)

            if self.weights == 'uniform':
                t = np.array([])
                for nn in neighbours:
                    t = np.append(t, nn[1])
                values, counts = np.unique(t, return_counts=True)
                y_pred.append(int(values[np.argmax(counts)]))

            else:
                yis = dict()
                t = np.array([])
                for nn in neighbours:
                    t = np.append(t, -nn[0])
                w = 1.0 / t
                # Get weights for each class
                for i in range(self.n_neighbors):
                    weight = w[i]
                    yi = neighbours[i][1]
                    if yi not in yis:
                        yis[yi] = weight
                    else:
                        yis[yi] += weight


                # Find yi with maximum weight
                max_w = 0
                for yi, weight in yis.items():
                    if max_w < weight:
                        max_w = weight
                        max_yi = yi

                y_pred.append(max_yi)
        return y_pred






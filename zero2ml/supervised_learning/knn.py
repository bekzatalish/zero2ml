from abc import ABC, abstractmethod
import numpy as np

from zero2ml.utils.distance_metrics import euclidian_distance
from zero2ml.utils.evaluation_metrics import Accuracy, RSquared

class KNN(ABC):
    """
    Class to represent a k-nearest neighbor model (classifier or regressor).
    """
    def __init__(self, model_type, k):
        """
        Create a k-nearest neighbor model.

        Parameters
        ----------
        model_type: str
            One of ['classifier', 'regressor'].
        k: int
            Number of neighbors.
        """
        self.model_type = model_type
        self.k = k

    def get_neighbors(self, train_data, test_row):
        """
        Get indices of k-nearest neighbors for a test sample.

        Parameters
        ----------
        train_data: array_like (m, n)
            Training features dataset with m examples and n features.
        test_row: array_like (n,)
            Testing sample with n features.

        Returns
        -------
        List of integer indices.
        """
        # Calculate distances between the testing sample and all training samples
        distances = list()
        for train_row in train_data:
            distances.append( euclidian_distance(test_row, train_row) )

        # Identify indices that would sort distances ascendingly
        indices = np.argsort(distances)

        # Select k number of neighbors
        indices = indices[0:self.k].tolist()

        return indices

    @abstractmethod
    def __predict_sample(self, target):
        """
        Predict target value given array of target values
        of nearest-neighbors.

        Parameters
        ----------
        values: array_like (m,)
            Array of target values.

        Returns
        -------
        Class prediction (if classifier) or floating point prediction (if regression).
        """
        pass

    def predict(self, X_train, y_train, X_test):
        """
        Predict given testing data using k-nearest neighbors of training data.

        Parameters
        ----------
        X_train: array_like (m, n)
            Training features dataset with m examples and n features.
        y_train: array_like (m,)
            Training target dataset with m examples.
        X_test: array_like (m, n)
            Testing features dataset with m examples and n features.

        Returns
        -------
        List of predictions.
        """
        # List to store predictions
        predictions = []

        # Make predictions for every test sample based on k-nearest neighbors
        for test_row in X_test:

            # Identify k-nearest neighbors of current test sample
            neighbors_indices = self.get_neighbors(X_train, test_row)

            # Identify target values of k-nearest neighbors
            neighbors_target = y_train[neighbors_indices]

            # Make prediction
            prediction = self.__predict_sample(neighbors_target)
            predictions.append(prediction)

        return predictions

    def score(self, X_train, y_train, X_test, y_test):
        """
        Calculate mean accuracy (for classification)
        or coefficient of determination (for regression)
        of predictions on a given data.

        Parameters
        ----------
        X_train: array_like (m, n)
            Training features dataset with m examples and n features.
        y_train: array_like (m,)
            Training target dataset with m examples.
        X_test: array_like (m, n)
            Testing features dataset with m examples and n features.
        y_train: array_like (m,)
            Testing target dataset with m examples.

        Returns
        -------
        Mean accuracy or coefficient of determination of predictions.
        """
        # Make predictions with the model
        y_pred = self.predict(X_train, y_train, X_test)

        if self.model_type == "classifier":

            # Calculate accuracy
            accuracy = Accuracy()
            score_value = accuracy(y_pred, y_test)

        if self.model_type == "regressor":

            # Calculate R^2
            R2 = RSquared()
            score_value = R2(y_pred, y_test)

        return score_value

class KNNClassifier(KNN):
    """
    Class to represent a k-nearest neighbor classifier.
    """
    def __init__(self, k):
        """
        Create a k-nearest neighbor classification model.

        Parameters
        ----------
        k: int
            Number of neighbors.
        """
        super().__init__(
            model_type="classifier",
            k=k
        )

    def _KNN__predict_sample(self, classes):
        """
        Predict target class given array of target classes
        of nearest-neighbors.

        Parameters
        ----------
        classes: array_like (m,)
            Array of target classes given as 0, 1, etc.

        Returns
        -------
        Class prediction.
        """
        # Calculate bin counts of k-nearest neighbors
        bin_counts = np.bincount(classes)

        # Make prediction for current test sample
        prediction = bin_counts.argmax()

        return prediction

class KNNRegressor(KNN):
    """
    Class to represent a k-nearest neighbor regressor.
    """
    def __init__(self, k):
        """
        Create a k-nearest neighbor regression model.

        Parameters
        ----------
        k: int
            Number of neighbors.
        """
        super().__init__(
            model_type="regressor",
            k=k
        )

    def _KNN__predict_sample(self, values):
        """
        Predict target value given array of target values
        of nearest-neighbors.

        Parameters
        ----------
        values: array_like (m,)
            Array of target values.

        Returns
        -------
        Floating point prediction.
        """
        # Calculate mean of k-nearest neighbors
        prediction = np.mean(values)

        return prediction

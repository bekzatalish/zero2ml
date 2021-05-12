import numpy as np
import math

from zero2ml.utils.evaluation_metrics import MeanSquaredError, RSquared
from zero2ml.utils.data_transformations import Standardize


class LinearRegression():
    """
    Multiple Linear Regression with input features standardization.

    Parameters
    ----------
    learning_rate: scalar
        Learning rate.

    Attributes
    ----------
    X: array_like
        Training features.
    y: array_like
        Training labels.
    samples_num: scalar
        Number of training samples.
    features_num: scalar
        Number of features.
    input_normalization: object
        Object used to standardize data and containing necessary parameters.
    self.W: array_like
        Weights.
    self.b: array_like
        Intercept term.
    self.training_loss: list
        List containing training loss for each iteration of model training.
    """
    def __init__(self, learning_rate=0.01):

        # Training data
        self.X = None

        # Training targets
        self.y = None

        # Number of training samples
        self.samples_num = None

        # Number of features
        self.features_num = None

        # Input features normalization
        self.input_normalization = Standardize()

        # Weights and intercept
        self.W = None
        self.b = None

        # Learning rate
        self.learning_rate = learning_rate

        # Loss and loss function
        self.training_loss = None
        self.__loss_function = MeanSquaredError()

    def fit(self, X, y, iterations=5000):
        """
        Fit the linear regression model with given training feature inputs.

        Parameters
        ----------
        X: array_like
            Training features.
        y: array_like
            Training labels.
        iterations: scalar
            Number of iterations for gradient descent to converge.
        """
        # Save X and y into model object
        self.X = X.copy()
        self.y = y.copy()

        # Apply normalization on features
        X_norm = self.input_normalization(self.X)

        # Calculate number of samples and features
        self.samples_num, self.features_num = self.X.shape

        # Initiate weights and intercept
        self.W = np.zeros(self.features_num)
        self.b = 0

        # Initiate loss
        self.training_loss = []

        # Gradient descent
        for iteration in range(iterations):

            # Make prediction
            pred = np.dot(X_norm, self.W) + self.b

            # Calculate loss and save to training loss list
            self.training_loss.append( self.__loss_function(pred, self.y) )

            # Calculate gradients
            dW = - ( 2 * np.dot(X_norm.T, self.y - pred) ) / self.samples_num
            db = - ( 2 * np.sum(self.y - pred) ) / self.samples_num

            # Update weights and intercept
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the label(s) with given feature inputs.

        Parameters
        ----------
        X: array_like
            Training features.

        Returns
        -------
        pred: array_like
            Predicted values.
        """
        # Apply Z-score normalization before
        X_norm = (X - self.input_normalization.mean) / self.input_normalization.std

        # Make predictions with the trained data
        pred = np.dot(X_norm, self.W) + self.b

        return pred

    def score(self, X, y):
        """
        Calculate coefficient of determination of predictions on a given data.

        Parameters
        ----------
        X: array_like
            Training features.
        y: array_like
            Training values.

        Returns
        -------
        Coefficient of determination of predictions.
        """
        # Make predictions with the trained model
        y_pred = self.predict(X)

        # Calculate mean squared error
        MSE = MeanSquaredError()
        mean_squared_error = MSE(y_pred, y)

        # Calculate R^2
        R2 = RSquared()
        r_squared = R2(y_pred, y)

        return r_squared

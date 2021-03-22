import numpy as np

from zero2ml.utils.activation_functions import Sigmoid
from zero2ml.utils.evaluation_metrics import CrossEntropyLoss, Accuracy


class LogisticRegression():
    """
    Logistic Regression classifier.

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

        # Weights and intercept
        self.W = None
        self.b = None

        # Sigmoid function
        self.__sigmoid = Sigmoid()

        # Learning rate
        self.learning_rate = learning_rate

        # Loss and loss function
        self.training_loss = None
        self.__loss_function = CrossEntropyLoss()

    def fit(self, X, y, iterations=5000):
        """
        Fit the logistic regression model with given training feature inputs.

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
            prob = self.__sigmoid( np.dot(self.X, self.W) + self.b)

            # Calculate loss and save to training loss list
            self.training_loss.append( self.__loss_function(prob, self.y) )

            # Calculate gradients
            dW = 1 / self.samples_num * np.dot(self.X.T, prob - self.y)
            db = 1 / self.samples_num * np.sum(prob - self.y)

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
            Label predictions. Could be 0 or 1.
        """
        # Calculate probabilities
        prob = self.__sigmoid( np.dot(X, self.W) + self.b )

        # Calculate actual predictions
        pred = (prob > 0.5).astype(int)

        return pred

    def score(self, X, y):
        """
        Calculates mean accuracy of predictions on a given data.

        Parameters
        ----------
        X: array_like
            Training features.
        y: array_like
            Training labels.

        Returns
        -------
        mean_accuracy: scalar
            Mean accuracy of predictions on a given data.
        """
        # Make predictions with the trained model
        y_pred = self.predict(X)

        # Calculate accuracy
        accuracy = Accuracy()
        mean_accuracy = accuracy(y_pred, y)

        return mean_accuracy

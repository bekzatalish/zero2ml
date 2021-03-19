import numpy as np


class CrossEntropyLoss():
    """
    Numerically stable Cross Entropy Loss (Logistic Loss) function.

    Reference: https://web.stanford.edu/~jurafsky/slp3/5.pdf
    Page 7, Formula 5.12

    Parameters
    ----------
    prob: array_like
        Predicted probabilities.
    y: array_like
        Actual labels.
    epsilon: scalar
        If any of probabilities values are 0 or 1, loss will be undefined, so epsilon is added inside the logs.

    Returns
    -------
    loss: scalar
        Cross Entropy Loss value of given predicted probabilities and actual labels.
    """
    def __call__(self, prob, y, epsilon=1e-15):
        loss = ( -( y * np.log(prob + epsilon) + (1 - y) * np.log(1 - prob + epsilon) )).mean()
        return loss

class Accuracy():
    """
    Calculates mean accuracy of predictions on a given data.

    Parameters
    ----------
    y_pred: array_like
        Predicted labels.
    y_true: array_like
        Actual labels.

    Returns
    -------
    mean_accuracy: scalar
        Mean accuracy of predictions on a given data.
    """
    def __call__(self, y_pred, y_true):
        accuracy = (y_pred == y_true).mean()
        return accuracy

class MeanSquaredError():
    """
    Calculates mean squared error of predictions on a given data.

    Parameters
    ----------
    y_pred: array_like
        Predicted values.
    y_true: array_like
        Actual values.

    Returns
    -------
    error: scalar
        Mean squared error of predictions on a given data.
    """
    def __call__(self, y_pred, y_true):
        error = np.abs(y_pred - y_true).mean()
        return error

class RSquared():
    """
    Calculate R^2.

    Parameters
    ----------
    y_pred: array_like
        Predicted values.
    y_true: array_like
        Actual values.

    Returns
    -------
    r_squared: scalar
        Calculated R^2.
    """
    def __call__(self, y_true, y_pred):

        # Calculate Pearson product-moment correlation coefficient
        correlation_matrix = np.corrcoef(y_true, y_pred)
        correlation = correlation_matrix[0,1]

        # Calculate R^2
        r_squared = correlation**2

        return r_squared

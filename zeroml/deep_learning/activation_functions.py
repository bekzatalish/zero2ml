import numpy as np


class Sigmoid():
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x: array_like
        Input value(s).
        
    Returns
    -------
    out: ndarray or scalar
        Element-wise sigmoid of x.
    """
    def _positive_sigmoid(self, x):
        out = 1 / (1 + np.exp(-x))
        return out

    def _negative_sigmoid(self, x):
        exp = np.exp(x)
        out = exp / (exp + 1)
        return out

    def __call__(self, x):

        # Identify positive and negative values
        positive_values = x >= 0
        negative_values = ~positive_values

        # Calculate numerically stable sigmoid of given array
        out = np.zeros(x.shape)
        out[positive_values] = self._positive_sigmoid(x[positive_values])
        out[negative_values] = self._negative_sigmoid(x[negative_values])

        return out

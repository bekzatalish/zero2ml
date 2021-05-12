import numpy as np

def euclidian_distance(a, b):
    """
    Compute Euclidian distance between two 1-D arrays.

    Parameters
    ----------
    a: array_like
        Input array.
    b: array_like
        Input array.

    Returns
    -------
    Euclidian distance between two vectors.
    """
    distance = 0.0

    for i in range(len(a)):
        distance += (a[i] - b[i])**2

    return np.sqrt(distance)

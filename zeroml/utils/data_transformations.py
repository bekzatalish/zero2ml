import numpy as np


def shuffle_data(X, y, random_state=None):
    """
    Shuffle X and y randomly or with a given seed.

    Parameters
    ----------
    X: array_like
        Training features.
    y: array_like
        Training labels.
    random_state: scalar
        Seed.

    Returns
    -------
    X, y: tuple of array_like
        Shuffled X and y.
    """
    # Set seed if needed
    if random_state is not None:
        np.random.seed(random_state)

    # Permute sequence of indices
    indices = np.random.permutation(X.shape[0])

    # Shuffle using permuted indices
    X, y = X[indices], y[indices]

    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split samples of X and y into train and test sets randomly or with a given seed.

    Parameters
    ----------
    X: array_like
        Training features.
    y: array_like
        Training labels.
    random_state: scalar
        Seed.
    shuffle: bool
        Whether or not to shuffle the data before splitting into train and test.

    Returns
    -------
    X_train, y_train, X_test, y_test: tuple of array_like
        Train and test sets of X and y.
    """
    # Shuffle if seed is provided
    if shuffle:
        X, y = shuffle_data(X, y, random_state=random_state)

    # Identify index on which to split
    samples_num = X.shape[0]
    test_size_num = int(X.shape[0] // (1 / test_size))
    split_index = samples_num - test_size_num

    # Split into train and test sets
    X_train, y_train= X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    return X_train, y_train, X_test, y_test

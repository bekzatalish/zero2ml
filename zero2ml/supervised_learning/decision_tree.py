from abc import ABC, abstractmethod
import numpy as np

from zero2ml.utils.evaluation_metrics import Accuracy, RSquared

class DecisionNode:
    """
    Class to represent a single node in a decision tree.
    """
    def __init__(self, left, right, decision_function, node_value=None):
        """
        Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to the left.

        Parameters
        ----------
        left: DecisionNode
            Left child node.
        right: DecisionNode
            Right child node.
        decision_function: func
            Function to decide on left or right node.
        node_value: int
            Value for leaf node. Default is None.
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.node_value = node_value

    def decide(self, input_data):
        """
        Get a child node based on the decision function.

        Parameters
        ----------
        input_data: array_like (m, n)
            Features dataset with shape m examples and n features.

        Returns
        -------
        Node value if a leaf node, otherwise a child node.
        """
        if self.node_value is not None:
            return self.node_value

        elif self.decision_function(input_data):
            return self.left.decide(input_data)

        else:
            return self.right.decide(input_data)

class DecisionTree(ABC):
    """
    Class to represent a decision tree (classifier or regressor).

    Attributes
    ----------
    root: DecisionNode
        Root node of the decision tree.
    """

    def __init__(self, model_type, splitting_criterion, max_depth):
        """
        Create a decision tree with a set depth limit.

        Starts with an empty root.

        Parameters
        ----------
        splitting_criterion: str
            The function to measure the quality of splits.
        model_type: str
            One of ['classifier', 'regressor'].
        max_depth: int
            The maximum depth to build the tree.
        """
        self.root = None
        self.model_type = model_type
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Build the tree from root using __build_tree().

        Parameters
        ----------
        X: array_like (m, n)
            Training features dataset with m examples and n features.
        y: array_like (m,)
            Training target dataset with m examples.
        """
        self.root = self.__build_tree(X, y)

    def __build_tree(self, X, y, depth=0):
        """
        Build tree that automatically finds the decision functions.

        Parameters
        ----------
        X: array_like (m, n)
            Features dataset with shape m examples and n features.
        y: array_like (m,)
            Target dataset with m examples.
        depth: int
            Depth to build tree to.

        Returns
        -------
        Root node of decision tree.
        """
        # Calculate number of training samples
        n_samples = X.shape[0]

        # For classifier, if all classes are the same,
        # then return a leaf node with the class label
        if self.model_type == "classifier":
            if np.unique(y).size == 1:
                return DecisionNode(None, None, None, y[0])

        # If the specified depth limit is reached, return a leaf with a node value
        if depth == self.max_depth:

            # Calculate node value
            # For classifier: most frequent class
            if self.model_type == "classifier":
                node_value = np.argmax(np.bincount(y.astype(np.int64)))
            # For regressor: mean
            if self.model_type == "regressor":
                node_value = np.mean(y, axis=0)

            # Return a leaf labeled with the most frequent class
            return DecisionNode(None, None, None, node_value)

        # Instantiate best impurity value
        best_impurity_gain = float("-inf")

        # Calculate initial impurity
        impurity = self.__compute_impurity(self.splitting_criterion, y)

        # Iterate over all possible columns and threshold values
        # to identify best splitting column and threshold
        for col in range(X.shape[1]):

            # Use arithmetic averages of boundary values as splitting thresholds
            # to avoid infinite value errors
            col_unique_values = np.unique(X[:,col])
            splitting_thresholds = (col_unique_values[:-1] + col_unique_values[1:]) / 2.0

            for threshold in splitting_thresholds:

                # Calculate impurity and sample ratio of left node
                y_left = y[X[:, col] <= threshold]
                impurity_left = self.__compute_impurity(self.splitting_criterion, y_left)
                sample_ratio_left = float(y_left.shape[0]) / n_samples

                # Calculate impurity and sample ratio of right node
                y_right = y[X[:, col] > threshold]
                impurity_right = self.__compute_impurity(self.splitting_criterion, y_right)
                sample_ratio_right = float(y_right.shape[0]) / n_samples

                # Calculate impurity gain as different between parent impurity
                # and weighted average impurity of child nodes
                impurity_gain = impurity - (sample_ratio_left * impurity_left + sample_ratio_right * impurity_right)

                # If current split is best, then save
                if impurity_gain > best_impurity_gain:
                    best_impurity_gain = impurity_gain
                    best_col = col
                    best_splitting_threshold = threshold

        # Split on best threshold
        root = DecisionNode(None, None, lambda feature: feature[best_col] <= best_splitting_threshold)

        # Left split
        left_indices = np.where(X[:,best_col] <= best_splitting_threshold)
        root.left = self.__build_tree(X[left_indices], y[left_indices], depth = depth + 1)

        # Right split
        right_indices = np.where(X[:,best_col] > best_splitting_threshold)
        root.right = self.__build_tree(X[right_indices], y[right_indices], depth = depth + 1)

        return root

    @abstractmethod
    def __compute_impurity(self, criterion, target):
        """
        Compute impurity for an array of target values.

        Parameters
        ----------
        criterion: str
            The function to measure the quality of splits.
        target: array_like (m,)
            Array of target values.
        """
        pass

    def predict(self, X):
        """
        Use the fitted tree to predict given data.

        Parameters
        ----------
        X: array_like (m, n)
            Dataset with m examples and n features.

        Returns
        -------
        A list of predictions.
        """
        predictions = []

        for row in X:
            predictions.append(self.root.decide(row))

        return predictions

    def score(self, X, y):
        """
        Calculate mean accuracy (for classification)
        or coefficient of determination (for regression)
        of predictions on a given data.

        Parameters
        ----------
        X: array_like (m, n)
            Features dataset with shape m examples and n features.
        y: array_like(m,)
            Target dataset with m examples.

        Returns
        -------
        Mean accuracy or coefficient of determination of predictions.
        """
        # Make predictions with the trained model
        y_pred = self.predict(X)

        if self.model_type == "classifier":

            # Calculate accuracy
            accuracy = Accuracy()
            score_value = accuracy(y_pred, y)

        if self.model_type == "regressor":

            # Calculate R^2
            R2 = RSquared()
            score_value = R2(y_pred, y)

        return score_value

class DecisionTreeClassifier(DecisionTree):
    """
    Class to represent a decision tree classifier.
    """
    def __init__(self, splitting_criterion="gini", max_depth=4):
        """
        splitting_criterion: str
            The function to measure the quality of splits.
        max_depth: int
            The maximum depth to build the tree.
        """
        super().__init__(
            model_type="classifier",
            splitting_criterion=splitting_criterion,
            max_depth=max_depth
        )

    def fit(self, X, y):
        """
        Build Decision Tree classifier.

        Parameters
        ----------
        X: array_like (m, n)
            Training features dataset with m examples and n features.
        y: array_like (m,)
            Training target dataset with m examples.
        """
        super().fit(X, y)

    def _DecisionTree__compute_impurity(self, criterion, target):
        """
        Compute impurity for an array of classes.

        Parameters
        ----------
        criterion: str
            The function to measure the quality of splits.
        target: array_like (m,)
            Array of target classes given as 0, 1, etc.
        """
        if criterion == 'gini':
            return gini_impurity(target)

        elif criterion == "entropy":
            return entropy(target)

        else:
            raise ValueError("Incorrect criterion specified. Cannot compute impurity.")

class DecisionTreeRegressor(DecisionTree):
    """
    Class to represent a decision tree regressor.
    """
    def __init__(self, splitting_criterion="variance", max_depth=4):
        """
        splitting_criterion: str
            The function to measure the quality of splits.
        max_depth: int
            The maximum depth to build the tree.
        """
        super().__init__(
            model_type="regressor",
            splitting_criterion=splitting_criterion,
            max_depth=max_depth
        )

    def fit(self, X, y):
        """
        Build Decision Tree regressor.

        Parameters
        ----------
        X: array_like (m, n)
            Training features dataset with m examples and n features.
        y: array_like (m,)
            Training target dataset with m examples.
        """
        super().fit(X, y)

    def _DecisionTree__compute_impurity(self, criterion, target):
        """
        Compute impurity for an array of target values.

        Parameters
        ----------
        criterion: str
            The function to measure the quality of splits.
        target: array_like (m,)
            Array of target values.
        """
        if criterion == 'variance':
            return variance(target)

        else:
            raise ValueError("Incorrect criterion specified. Cannot compute impurity.")

def gini_impurity(classes):
    """
    Compute the gini impurity for an array of classes.

    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels.
    It reaches its minimum at zero when classes are the same.

    Parameters
    ----------
    classes: array_like (m,)
        Array of classes given as 0, 1, etc.

    Returns
    -------
    Floating point number representing the gini impurity.
    """
    # Calculate number of samples
    n_samples = classes.size

    # Handle case of zero number of samples
    if n_samples == 0:
        return 0.0

    # Identify unique classes
    unique_classes = np.unique(classes)

    # Calculate probabilities for each class
    probabilities = [np.sum(classes == class_val) / n_samples for class_val in unique_classes]

    # Calculate gini impurity
    gini_impurity = 1.0 - np.sum([p**2 for p in probabilities])

    return gini_impurity

def entropy(classes):
    """
    Compute entropy for an array of classes.

    Parameters
    ----------
    classes: array_like (m,)
        Array of classes given as 0, 1, etc.

    Returns
    -------
    Floating point number representing the entropy.
    """
    # Calculate number of samples
    n_samples = classes.size

    # Handle case of zero number of samples
    if n_samples == 0:
        return 0.0

    # Identify unique classes
    unique_classes = np.unique(classes)

    # Calculate probabilities for each class
    probabilities = [np.sum(classes == class_val) / n_samples for class_val in unique_classes]

    # Calculate entropy
    entropy = np.sum([-p * np.log2(p) for p in probabilities])

    return entropy

def variance(values):
    """
    Compute the variance for an array of target values.

    Parameters
    ----------
    values: array_like (m,)
        Array of target values.

    Returns
    -------
    Floating point number representing the variance.
    """
    # Calculate mean
    mean = np.mean(values)

    # Calculate variance
    variance = np.mean( (values - mean)**2 )

    return variance

import os
import numpy as np

from zero2ml.utils.data_transformations import train_test_split
from zero2ml.supervised_learning.decision_tree import DecisionTreeRegressor


def main():

    # Construct path to dataset
    root_directory_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = os.path.join(root_directory_path, "tests", "test_data", "housing.csv")

    # Read dataset
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:,:-1]
    y = data[:,-1].astype(float)

    # Train test split
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Instantiate model
    model = DecisionTreeRegressor(max_depth=2)

    # Fit model
    model.fit(X_train, y_train)

    # Calculate train and test R^2
    train_results = model.score(X_train, y_train)
    test_results = model.score(X_test, y_test)

    print("Finished training Regression Tree model.\n")
    print("Training R^2: {:0.6f}".format(test_results))
    print("Testing R^2: {:0.6f}".format(test_results))

if __name__ == "__main__":
    main()

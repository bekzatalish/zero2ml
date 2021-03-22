import os
import numpy as np

from zero2ml.utils.data_transformations import train_test_split
from zero2ml.supervised_learning.regression.linear_regression import LinearRegression


def main():

    # Construct path to dataset
    root_directory_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = os.path.join(root_directory_path, "tests", "test_data", "housing.csv")

    # Read dataset
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:,:-1]
    y = data[:,-1].astype(int)

    # Train test split
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Instantiate model
    model = LinearRegression(learning_rate=0.01)

    # Fit model
    model.fit(X_train, y_train)

    # Calculate train and test accurracy
    train_metrics = model.score(X_train, y_train)
    test_metrics = model.score(X_test, y_test)

    print("Finished training Linear Regression model.\n")
    print("Training MSE: {:0.3f}".format(train_metrics["Mean Squared Error"]))
    print("Testing MSE: {:0.3f}\n".format(test_metrics["Mean Squared Error"]))
    print("Training R^2: {:0.3f}".format(train_metrics["R Squared"]))
    print("Testing R^2: {:0.3f}".format(test_metrics["R Squared"]))

if __name__ == "__main__":
    main()

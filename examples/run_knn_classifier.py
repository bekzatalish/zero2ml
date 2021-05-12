import os
import numpy as np

from zero2ml.utils.data_transformations import train_test_split
from zero2ml.supervised_learning.knn import KNNClassifier


def main():

    # Construct path to dataset
    root_directory_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = os.path.join(root_directory_path, "tests", "test_data", "breast_cancer.csv")

    # Read dataset
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:,:-1]
    y = data[:,-1].astype(int)

    # Train test split
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1, random_state=21)

    # Instantiate model
    model = KNNClassifier(k=5)

    # Calculate test accuracy
    test_results = model.score(X_train, y_train, X_test, y_test)

    print("Finished training k-nearest neighbor classification model.\n")
    print("Testing accuracy: {:0.6f}".format(test_results))

if __name__ == "__main__":
    main()

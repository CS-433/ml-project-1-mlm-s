import numpy as np

from helpers import *
from implementations import *
from helper_functions import *
from data_cleaning import *
from data_processing import *
import cross_validation as cv


def main():

    # Load data paths
    TRAIN_PATH = "./dataset/train.csv"
    TEST_PATH = "./dataset/test.csv"

    y_train, x_train, ids = load_csv_data(TRAIN_PATH)
    y_test, x_test, ids_test = load_csv_data(TEST_PATH)

    y = y_train.copy()
    tX_train, masks_train = cleaning(x_train)
    tX_test, masks_test = cleaning(x_test)

    # Initialize predictions
    prediction = np.zeros(tX_test.shape[0])
    accuracies = []
    k_fold = 10
    seed = 10
    weights = []

    # Selected parameters
    # degree = [6, 6, 8]
    # lambda_ = [1e-7, 1e-6, 1e-5]
    degree = [9, 9, 9]
    lambda_ = [1e-5, 1e-5, 1e-5]

    for i in range(len(masks_train)):

        train_data = tX_train[masks_train[i]]
        train_y = y[masks_train[i]]
        test_data = tX_test[masks_test[i]]

        train_data = process_data(train_data)
        test_data = process_data(test_data)

        # Build poly
        train_phi = build_polynomial(train_data, degree[i])
        test_phi = build_polynomial(test_data, degree[i])

        # Obtain weight
        weight, _ = ridge_regression(train_y, train_phi, lambda_[i])

        # Generate predictions
        pred_y = predict_labels(weight, test_phi)
        prediction[masks_test[i]] = pred_y

        k_indices = cv.build_k_indices(train_y, k_fold, seed)

        for k in range(k_fold):
            accuracy = cv.cross_validation(
                train_y, train_phi, k_indices, k, ridge_regression, lambda_=lambda_[i]
            )
            accuracies.append(accuracy)

    # Generate file csv for submission
    OUTPUT_PATH = "./dataset/submission.csv"
    create_csv_submission(ids_test, prediction, OUTPUT_PATH)
    print("Submission file created!")

    print("Accuracy for this model is:", np.mean(accuracies))
    print("Std for this model is:", np.std(accuracies))


if __name__ == "__main__":
    main()

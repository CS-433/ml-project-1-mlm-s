import numpy as np


def standardization(X_clean):
    # standardize along the columns
    for i in range(X_clean.shape[1]):
        X_clean[:, i] = (X_clean[:, i] - np.mean(X_clean[:, i])) / np.std(X_clean[:, i])
    return X_clean


def replace_missing_by_median(X):
    # Replace -999 by Nan
    X[X == -999] = np.nan
    #  Replace nan by median of the column
    for i in range(X.shape[1]):
        median = np.nanmedian(X[:, i])
        X[np.isnan(X[:, i]), i] = median
    return X


def process_data(X):
    X = replace_missing_by_median(X)
    X = standardization(X)
    X = np.nan_to_num(X)
    return X

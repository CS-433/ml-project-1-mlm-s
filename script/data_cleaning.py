import numpy as np


def drop_features(X, features=[5, 6, 12, 21, 24, 25, 26, 27, 28, 29]):
    """
    Drop features with cov>0.9
    """
    np.delete(X, features, 1)
    return X


def replace_outliers_with_nan(X, keep=0.95):
    """
    Replace outliers with nan.

    :param x: x matrix.
    :param keep: Percentile of values to be kept from the x matrix.
    :return: x matrix with nan values in place of outliers.
    """
    for i in range(X.shape[1]):
        min_value = np.quantile(X[:, i], (1 - keep) / 2)
        max_value = np.quantile(X[:, i], (1 + keep) / 2)
        values_to_be_changed = np.logical_or(X[:, i] < min_value, X[:, i] > max_value)
        X[values_to_be_changed, i] = np.nan
    return X


def JET_num_mask(
    X,
):  # Checked and correct. There is 2 and 3, but there is also number greater than 3, should we drop them or not ?
    mask_0 = X[:, 22] == 0
    mask_1 = X[:, 22] == 1
    mask_2 = X[:, 22] >= 2
    return [mask_0, mask_1, mask_2]


def cleaning(X, features=[5, 6, 12, 21, 24, 25, 26, 27, 28, 29], keep=0.95):
    X = drop_features(X, features)
    X = replace_outliers_with_nan(X, keep)
    mask = JET_num_mask(X)
    return X, mask

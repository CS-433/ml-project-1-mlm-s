from cmath import e
import numpy as np


def e_vector(y, tx, w):
    """Compute the error vector e used for other computations
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
           the value of the error vector e"""

    e = y - np.dot(tx, w)
    return e


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    # compute loss by MSE
    e = e_vector(y, tx, w)
    mse = 1 / 2 * np.mean(e**2)
    return mse


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = e_vector(y, tx, w)
    # compute gradient vector
    gradient = -tx.T.dot(e) / len(e)
    return gradient


def compute_stoch_gradient(y, tx, w):
    """Computes the stochastic gradient for few samples.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = e_vector(y, tx, w)
    stoch_gradient = -tx.T.dot(e) / len(e)
    return stoch_gradient


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """apply the sigmoid function on t.
    Args:
        t: scalar or numpy array

    Returns:
        sigmoid function (scalar or numpy array)
    """
    sigmoid = 1.0 / (1 + np.exp(-t))
    return sigmoid


def compute_log_loss(y, tx, w):
    """Calculate the loss using negative log likelihood.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        the value of the non-negative loss corresponding to the inputs.
    """
    logistic_function = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(logistic_function)) + (1 - y).T.dot(
        np.log(1 - logistic_function)
    )
    return np.squeeze(-loss).item() / len(y)


def compute_log_gradient(y, tx, w):
    """Computes the gradient at w using the logistic loss.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the logistic gradient of the loss at w.
    """
    logistic_function = sigmoid(tx.dot(w))
    log_gradient = tx.T.dot(logistic_function - y) / len(y)
    return log_gradient

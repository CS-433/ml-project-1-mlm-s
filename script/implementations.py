import numpy as np
from helper_functions import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear Regression using gradient descent:
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        (w, loss) : the last weight vector of the iteration and its corresponding loss value
    """
    # Define treshold
    threshold = 1e-8

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):

        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        if np.linalg.norm(gradient) < threshold:
            break  # convergence criterion met

    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear Regression using stochastic gradient descent:
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        batch_
        gamma: a scalar denoting the stepsize

    Returns:
        (w, loss) : the last weight vector of the iteration and its corresponding loss value
    """
    # Define treshold
    threshold = 1e-8

    # Define w and loss at step 0
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):

            # compute a stochastic gradient and loss
            stoch_gradient = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_mse(y, tx, w)
            # update w through the stochastic gradient
            w = w - gamma * stoch_gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution using normal equations.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w[-1]: the optimal weight vector
        loss: corresponding MSE loss (a scalar).
    """

    # least squares:
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)

    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w[-1]: the optimal weight vector
        loss: corresponding MSE loss (a scalar).
    """
    a = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + a
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)

    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic Regression using gradient descent.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        (w, losses) : the last weight vector of the iteration and its corresponding loss value
    """
    # treshold init
    threshold = 1e-8

    # Define w and loss at step 0
    w = initial_w
    loss = np.Infinity

    for n_iter in range(max_iters):
        # Compute gradient and loss
        log_gradient = compute_log_gradient(y, tx, w)
        loss = compute_log_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * log_gradient

        if np.linalg.norm(log_gradient) < threshold:
            break  # convergence criterion met

    loss = compute_log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        (w[-1], losses[-1]) : the last weight vector of the iteration and its corresponding loss value after regularization
    """

    threshold = 1e-8
    # Define w and loss at step 0
    w = initial_w
    loss = np.Infinity

    for n_iter in range(max_iters):
        # Compute gradient and loss
        log_gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
        loss_reg = compute_log_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * log_gradient

        if np.linalg.norm(log_gradient) < threshold:
            break  # convergence criterion met

    loss_reg = compute_log_loss(y, tx, w)
    return w, loss_reg

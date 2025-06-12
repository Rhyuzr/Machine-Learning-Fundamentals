import numpy as np


def accuracy(y_hat, y):
    """
    Calculates the accuracy of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The accuracy as a float between 0 and 1.
    """
    return np.mean(y_hat == y)


def precision(y_hat, y):
    """
    Calculates the precision of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The precision as a float between 0 and 1.
    """
    true_positive = np.sum((y_hat == 1) & (y == 1))
    false_positive = np.sum((y_hat == 1) & (y == 0))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0


def recall(y_hat, y):
    """
    Calculates the recall of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The recall as a float between 0 and 1.
    """
    true_positive = np.sum((y_hat == 1) & (y == 1))
    false_negative = np.sum((y_hat == 0) & (y == 1))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0


def f1_score_metric(y_hat, y):
    """
    Calculates the F1-score of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The F1-score as a float between 0 and 1.
    """
    p = precision(y_hat, y)
    r = recall(y_hat, y)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def mse(y_hat, y):
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The MSE.
    """
    return np.mean((y_hat - y) ** 2)


def rmse(y_hat, y):
    """
    Calculates the Root Mean Squared Error (RMSE).

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The RMSE.
    """
    return np.sqrt(mse(y_hat, y))


def mae(y_hat, y):
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The MAE.
    """
    return np.mean(np.abs(y_hat - y))


def r2_score(y_hat, y):
    """
    Calculates the R-squared score.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The R-squared score.
    """
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_hat) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
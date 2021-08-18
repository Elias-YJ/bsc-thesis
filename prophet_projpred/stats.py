import numpy as np


def elpd(loglik: np.array, weights=1, test_indices=None):
    """Calculates the elpd score, given log likelihood matrix and weights.
    Let N be the number of data points and S the number of draws

    :param loglik: NxS matrix
    :param weights:
    :param test_indices: array(bool) Nx1
    :return:
    """
    w = loglik + np.log(weights)
    max_w = np.amax(loglik)
    lppd = max_w + np.log(np.sum(np.exp(w-max_w), axis=1))
    if test_indices is not None:
        return np.sum(lppd[test_indices])
    else:
        return np.sum(lppd)


def mape(y, predictions, test_indices=None):
    """

    :param y:
    :param predictions:
    :param test_indices:
    :return:
    """
    if test_indices is not None:
        value = np.mean(
            np.mean(np.abs((y - predictions) / y), axis=1)[test_indices]
        )
    else:
        value = np.mean(np.mean(np.abs((y - predictions) / y), axis=1))
    return value

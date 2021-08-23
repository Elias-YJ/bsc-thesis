import numpy as np


def lppd(loglik, weights=1):
    w = loglik + np.log(weights)
    max_w = np.amax(loglik)
    result = max_w + np.log(np.sum(np.exp(w - max_w), axis=1))
    return result


def elpd(loglik, weights=1, indices=None):
    """Calculates the elpd score, given log likelihood matrix and weights.
    Let N be the number of data points and S the number of draws

    :param loglik: NxS matrix
    :param weights:
    :param indices: array(bool) Nx1
    :return:
    """
    if indices is not None:
        return np.sum(lppd(loglik, weights)[indices])
    else:
        return np.sum(lppd(loglik, weights))


def elpd_se(loglik, weights=1, indices=None):
    lppd_values = lppd(loglik, weights)
    if indices is not None:
        lppd_values = lppd_values[indices]
    lppd_mean = np.mean(lppd_values)
    n = len(lppd_values)
    sd = np.sqrt(
        np.sum(
            weights*(lppd_values-lppd_mean)**2/np.sum(weights)
        )/(n-1)
    )
    se = n*sd/np.sqrt(n)
    return se


def mape(y, predictions, indices=None):
    """

    :param y:
    :param predictions:
    :param indices:
    :return:
    """
    if indices is not None:
        value = np.mean(
            np.mean(np.abs((y - predictions) / y), axis=1)[indices]
        )
    else:
        value = np.mean(np.mean(np.abs((y - predictions) / y), axis=1))
    return value

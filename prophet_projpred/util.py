import numpy as np


def thinning(n, n_total):
    return range(
        0,
        n_total,
        max(1, int(np.floor(n_total / n)))
    )

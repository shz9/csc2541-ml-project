"""

Author: Shadi Zabad
Date: December 2017

This script implements the Block Bootstrap (BB) of Kunsch (1989) and
the Continuous-Path Block Bootstrap (CBB) of Paparoditis & Politis (2001).

Implementations below are based on algorithms described in:

PAPARODITIS & POLITIS 2001
The Continuous-Path Block-Bootstrap

http://www.math.ucsd.edu/~politis/PAPER/CBBfest.pdf

------------------------------------------------------------------------

NOTE: The CBB implementation doesn't seem as robust as I'd like, maybe
there's an error in the logic somewhere. I tested it with non-stationary
data (e.g. [x + np.sin(x) for x in range(100)]) and the samples I got
don't have the linear trend. But they're continuous.

"""

import numpy as np


def block_bootstrap(ts_data, window_size):

    n = len(ts_data)
    b = window_size
    k = n / b

    if window_size >= n:
        raise ValueError("Window size has to be less than the length of the dataset!")

    block_ids = list(range(n - b + 1))

    # Selected block IDs:
    i_ = [np.random.choice(block_ids) for _ in range(k)]

    x_star = []

    for m in range(k):
        for j in range(window_size):
            x_star.append(ts_data[i_[m] + j - 1])

    return x_star


def continuous_path_block_bootstrap(ts_data, window_size):

    n = len(ts_data)
    b = window_size
    k = n / b

    # First calculate centered residuals:
    u_t = []

    for t in range(1, n):
        u_t.append(ts_data[t] - ts_data[t - 1])

    sub_term = float(sum(u_t)) / (n - 1)

    for t in range(len(u_t)):
        u_t[t] -= sub_term

    # For t = 0, x_telda[t] = x[t]:
    x_telda = [ts_data[0]]

    # For t >= 1:
    for t in range(1, n):
        x_telda.append(ts_data[0] + sum(u_t[:t]))

    block_ids = list(range(n - b))

    # Selected block IDs:
    i_ = [np.random.choice(block_ids) for _ in range(k)]

    x_star = []

    # Construction of the first block:

    for j in range(b):
        x_star.append(
            ts_data[0] + x_telda[i_[0] + j] - x_telda[i_[0]]
        )

    # The remaining blocks:

    for m in range(k - 1):
        for j in range(b):
            x_star.append(
                x_star[m * b] + x_telda[i_[m] + j] - x_telda[i_[m]]
            )

    return x_star

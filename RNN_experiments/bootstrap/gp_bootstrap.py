"""
Author: Shadi Zabad
Date: November 2017

This script uses Gaussian Processes to create sub-samples to train RNNs on.
** Work in progress... not sure it's producing the desired behavior. **
"""

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


def bootstrap_data(X, Y, kernel, n_samples=50):
    """

    :param X:
    :param Y:
    :param kernel:
    :param samples:
    :return:
    """

    # First, we fit a GP model to the data:

    m = GaussianProcessRegressor(kernel=kernel)
    m.fit(X, Y)

    boot_samples = []
    num_records = len(X)

    for si in range(n_samples):
        # - Start of the sequence is a normal distribution centered on X[0]
        # and with standard deviation equal to 5% the value of X[0].
        # - Same logic applies to the end of the sequence.
        # - The number of steps in the sequence is equal to the number
        # of records in X +/- 10% of that number.
        xx = np.atleast_2d(np.linspace(np.random.normal(loc=X[0], scale=.0001 * X[0]),
                                       np.random.normal(loc=X[-1], scale=.0001 * X[-1]),
                                       num_records + np.random.randint(low=-int(.1 * num_records),
                                                                       high=int(.1 * num_records)))).T
        yy = m.predict(xx)
        boot_samples.append((xx, yy))

    return boot_samples

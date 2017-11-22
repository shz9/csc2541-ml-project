"""
Author: Shadi Zabad
Date: November 2017

This script uses Gaussian Processes to create sub-samples to train RNNs on.
** Work in progress... not sure it's producing the desired behavior. **
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import pandas as pd
import pickle
import os
import glob
import numpy as np


def bootstrap_data(X, Y, kernel, n_samples=50):

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
        yy = m.predict(xx, )
        boot_samples.append((xx, yy))

    return boot_samples


def generate_samples(n_samples):

    ex_dataset = pd.read_csv('../../../data/mauna-loa-atmospheric-co2.csv',
                             header=None)
    ex_dataset.columns = ['CO2Concentration', 'Time']

    train_data = ex_dataset.loc[ex_dataset.Time <= 1980, ['CO2Concentration', 'Time']]

    samps = bootstrap_data(train_data['Time'].reshape(-1, 1),
                           train_data['CO2Concentration'].reshape(-1, 1),
                           34.4 ** 2 * RBF(length_scale=41.8) +
                           3.27 ** 2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                                              periodicity=1) +
                           0.446 ** 2 * RationalQuadratic(alpha=17.7, length_scale=0.957) +
                           0.197 ** 2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336),
                           n_samples=n_samples)

    for osf in glob.glob("./gp_samples/*.pkl"):
        os.remove(osf)

    for idx, samp in enumerate(samps):
        with open("./gp_samples/" + str(idx) + ".pkl", "wb") as sf:
            pickle.dump(samp, sf)


if __name__ == '__main__':
    generate_samples(100)

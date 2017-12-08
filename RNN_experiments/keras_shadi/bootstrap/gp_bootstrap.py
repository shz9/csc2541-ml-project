"""
Author: Shadi Zabad
Date: November 2017

This script uses Gaussian Processes to create sub-samples to train RNNs on.
** Work in progress... not sure it's producing the desired behavior. **
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from data.data_reader import read_mauna_loa_co2_data, read_lake_erie_data
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import pandas as pd
import pickle
import os
from matplotlib import pylab as plt
import glob
import numpy as np


def gp_bootstrap_shift(X, Y, kernel, n_samples=50):

    # First, we fit a GP model to the data:

    m = GaussianProcessRegressor(kernel=kernel, alpha=20)
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


def gp_bootstrap_noise(X, Y, kernel, n_samples=50, alpha=20.0):

    # First, we fit a GP model to the data
    # (with noise added, represented by the parameter alpha):
    m = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    m.fit(X, Y)

    gp_samples = m.sample_y(X, n_samples=n_samples).T

    boot_samples = []

    for si in range(n_samples):
        boot_samples.append((X, gp_samples[si][0]))

    return boot_samples


def generate_co2_samples(n_samples):

    co2_dataset = read_mauna_loa_co2_data()

    train_data = co2_dataset.iloc[:int(.7 * len(co2_dataset)), ]

    samps = gp_bootstrap_noise(train_data['Time'].reshape(-1, 1),
                               train_data['CO2Concentration'].reshape(-1, 1),
                               34.4 ** 2 * RBF(length_scale=41.8) +
                               3.27 ** 2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                                                  periodicity=1) +
                               0.446 ** 2 * RationalQuadratic(alpha=17.7, length_scale=0.957) +
                               0.197 ** 2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336),
                               n_samples=n_samples,
                               alpha=19.0)

    for osf in glob.glob("./gp_samples/co2/*.pkl"):
        os.remove(osf)

    for idx, samp in enumerate(samps):
        with open("./gp_samples/co2/" + str(idx) + ".pkl", "wb") as sf:
            pickle.dump(samp, sf)


def generate_erie_samples(n_samples):
    erie_dataset = read_lake_erie_data()

    train_data = erie_dataset.iloc[:int(.7 * len(erie_dataset)), ]

    samps = gp_bootstrap_noise(np.array(train_data.index).reshape(-1, 1),
                               train_data['Level'].reshape(-1, 1),
                               3.27 ** 2 * ExpSineSquared(length_scale=180, periodicity=10) *
                               ExpSineSquared(length_scale=1.44),
                               n_samples=n_samples,
                               alpha=.09)

    for osf in glob.glob("./gp_samples/erie/*.pkl"):
        os.remove(osf)

    for idx, samp in enumerate(samps):
        with open("./gp_samples/erie/" + str(idx) + ".pkl", "wb") as sf:
            pickle.dump(samp, sf)


def plot_samples(dataset, limit=None):

    samp_dat = []

    for s in glob.glob("./gp_samples/" + dataset + "/*.pkl"):

        with open(s, "rb") as sf:
            samp_dat.append(pickle.load(sf))

        if limit is not None:
            if len(samp_dat) >= limit:
                break

    for dat in samp_dat:
        plt.plot(dat[1], alpha=.3)

    plt.show()


def test():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dataset = read_lake_erie_data()
    v = np.array(dataset.index).reshape(-1, 1)
    bootstrapped_dataset = gp_bootstrap_noise(np.array(dataset.index).reshape(-1, 1),
                                              dataset["Level"].reshape(-1, 1),
                                              3.27 ** 2 * ExpSineSquared(length_scale=180, periodicity=10) *
                                              ExpSineSquared(length_scale=1.44),
                                              n_samples=50,
                                              alpha=.09)

    co2_dataset = read_mauna_loa_co2_data()

    samps = gp_bootstrap_noise(co2_dataset['Time'].reshape(-1, 1),
                               co2_dataset['CO2Concentration'].reshape(-1, 1),
                               34.4 ** 2 * RBF(length_scale=41.8) +
                               3.27 ** 2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                                                  periodicity=1) +
                               0.446 ** 2 * RationalQuadratic(alpha=17.7, length_scale=0.957) +
                               0.197 ** 2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336),
                               n_samples=50,
                               alpha=30.0)

    f, (ax1, ax2) = plt.subplots(1, 2)

    for dat in bootstrapped_dataset:
        ax1.plot(dat[1], alpha=.1, color="gray")

    ax1.plot(dataset["Level"], color="blue")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Level (M)")

    for dat in samps:
        ax2.plot(dat[1], alpha=.1, color="gray")

    ax2.plot(co2_dataset['CO2Concentration'], color="blue")

    ax2.set_xlabel("Month")
    ax2.set_ylabel("CO2 Concentration (PPM)")

    plt.suptitle("Example of Samples from GP Bootstrap")

    plt.show()


if __name__ == '__main__':
    #test()
    #generate_co2_samples(100)
    #generate_erie_samples(100)
    plot_samples("co2", limit=50)

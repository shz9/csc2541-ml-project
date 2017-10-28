"""
The functions below are borrowed from the GPflow documentation.
http://gpflow.readthedocs.io/en/latest/notebooks/kernels.html
"""

import numpy as np
import matplotlib.pyplot as plt


def plotkernelsample(k, ax, k_name=None, xmin=-3, xmax=3):

    xx = np.linspace(xmin, xmax, 100)[:, None]
    K = k.compute_K_symm(xx)
    K = np.nan_to_num(K)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)

    if k_name is None:
        ax.set_title(k.__class__.__name__)
    else:
        ax.set_title(k_name)


def plotkernelfunction(k, ax, k_name=None, xmin=-3, xmax=3, other=0):

    xx = np.linspace(xmin, xmax, 100)[:, None]
    K = k.compute_K_symm(xx)
    ax.plot(xx, k.compute_K(xx, np.zeros((1, 1)) + other))
    if k_name is None:
        ax.set_title(k.__class__.__name__ + ' k(x, %f)' % other)
    else:
        ax.set_title(k_name + ' k(x, %f)' % other)


def plot_kernels(kernel_list, nrows=2, ncols=4):

    f, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    kernel_list = kernel_list[::-1]
    num_plots = len(kernel_list) - 1

    for r in range(nrows):

        for c in range(ncols):
            plotkernelsample(kernel_list[num_plots]["Function"], axes[r, c], kernel_list[num_plots]["Name"])
            num_plots -= 1
            if num_plots < 0:
                break
        else:
            continue

        break

    plt.show()

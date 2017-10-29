import numpy as np
from matplotlib import pyplot as plt


def gp_plot(plot_path, x, y, gp_mean=None, mean_color="b", var_color="b", point_color="black"):
    """
    This function was borrowed from the GPflow tutorial. Modified by Shadi Zabad.
    http://gpflow.readthedocs.io/en/latest/notebooks/regression.html
    """

    fig = plt.figure(figsize=(12, 6))

    if x is not None and y is not None:
        plt.plot(x, y, 'kx', color=point_color, mew=2)

    if gp_mean is not None:
        plt.plot(gp_mean["X"], gp_mean["Y"], color=mean_color, lw=2)
        plt.fill_between(gp_mean["X"][:, 0],
                         gp_mean["Y"][:, 0] - 2*np.sqrt(gp_mean["Var"][:, 0]),
                         gp_mean["Y"][:, 0] + 2*np.sqrt(gp_mean["Var"][:, 0]),
                         color=var_color, alpha=0.2)

    fig.savefig(plot_path)


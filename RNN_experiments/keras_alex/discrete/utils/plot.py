import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(x, true, pred, title, file_path, params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax1.plot(np.arange(x.shape[0]), pred.T, color="r", label="Predicted", alpha=1)
    ax1.set_title(title)
    ax1.legend()

    param_string = create_param_string(params)

    ax2.text(0.1, 0.5, param_string, horizontalalignment="left", verticalalignment="center",
            transform=ax2.transAxes, wrap=True)


    fig.savefig(file_path)


def plot_error(x, true, mean, std, title, file_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax.plot(np.arange(x.shape[0]), mean.reshape(-1), color="r", label="Pred")
    ax.fill_between(np.arange(x.shape[0]), (mean + (2 * std)).reshape(-1), (mean - ( 2* std)).reshape(-1), color="r", alpha=0.2)
    ax.legend()
    ax.set_title(title)

    fig.savefig(file_path)


def plot_residuals(x, true, pred, title, file_path, params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.scatter(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax1.scatter(np.arange(x.shape[0]), pred.reshape(-1), color="r",  label="Pred")
    ax1.legend()
    ax1.set_title(title)

    param_string = create_param_string(params)
    ax2.set_title("Params")
    ax2.text(0.1, 0.5, param_string, horizontalalignment="left", verticalalignment="center", transform=ax2.transAxes,
             wrap=True)

    fig.savefig(file_path)


def create_param_string(params):
    param_string = ""
    primitive = (int, str, bool, float)

    for key, value in params.iteritems():
        if not isinstance(value, primitive):
            value = str(type(value)) + str(value.get_config())

        param_string += key + ": " + str(value)
        param_string += "\n"

    return param_string
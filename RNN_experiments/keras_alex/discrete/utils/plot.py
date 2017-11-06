import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(x, true, pred, title, file_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax.plot(np.arange(x.shape[0]), pred.reshape(-1), color="r", label="Predicted")
    ax.set_title(title)
    ax.legend()

    fig.savefig(file_path)
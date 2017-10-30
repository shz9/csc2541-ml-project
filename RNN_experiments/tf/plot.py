import matplotlib.pyplot as plt
import numpy as np


def format_predictions(y_pred, seq_len, seq_dim):
    z = np.array(y_pred)

    z = np.transpose(z, [0, 2, 1, 3])
    z = z.reshape(z.shape[0] * z.shape[1], seq_len, seq_dim)

    return z


def plot_predictions(x, true, pred):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(x[0: x.shape[0], x.shape[1] - 1, ].reshape(-1), true[0: true.shape[0], true.shape[1] - 1, ].reshape(-1), color="b")
    ax.plot(x[0: x.shape[0], x.shape[1] - 1, ].reshape(-1), pred[0: pred.shape[0], pred.shape[1] - 1, ].reshape(-1), color="r")

    plt.show()
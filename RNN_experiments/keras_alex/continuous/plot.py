import matplotlib.pyplot as plt


def plot_predictions(x, true, pred):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(x[:, :, 0].reshape(-1), true.reshape(-1))
    ax.plot(x[:, :, 0].reshape(-1), pred.reshape(-1))

    plt.show()


def plot_predictions_window(x, true, pred):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(x[0: x.shape[0], x.shape[1] - 1, 0].reshape(-1), true.reshape(-1), color="b")
    ax.plot(x[0: x.shape[0], x.shape[1] - 1, 0].reshape(-1), pred.reshape(-1), color="r")

    plt.show()


def plot_temp(x, true, pred):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(x, true, color="b")
    ax.plot(x, pred, color="r")

    plt.show()
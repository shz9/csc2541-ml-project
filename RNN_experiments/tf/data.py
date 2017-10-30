import numpy as np


def generate_data(samples=100, seq_len=50, seq_dim=1, fun=np.sin):
    x_train = []
    temp = np.linspace(-10, 10, (samples + (seq_len - 1)) * 10)

    for i in range(len(temp) - (seq_len)):
        x_train.append(temp[i: i + seq_len])

    x_train = np.array(x_train)
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], 1))
    y_train = fun(x_train)

    x_test = []
    temp = np.linspace(30, 50, (samples + (seq_len - 1)) * 10)

    for i in range(len(temp) - (seq_len)):
        x_test.append(temp[i: i + seq_len])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[1], 1))
    y_test = fun(x_test)

    return x_train, y_train, x_test, y_test


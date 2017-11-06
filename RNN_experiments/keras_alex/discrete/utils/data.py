import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_co2(file_path):
    co2 = pd.read_csv(file_path, header=None, index_col=None)

    return co2


def create_window_data(data, window_size):
    temp = []

    for i in range(len(data) - window_size):
        temp.append(data[i: i + window_size, ])

    temp = np.array(temp)

    x = temp[:, :-1]
    y = temp[:, -1]

    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=1)

    return x, y


def scale_data(x_window, y_window, train_percent):
    idxs = np.arange(x_window.shape[0])
    train_idx = idxs[:int(train_percent * x_window.shape[0])]
    test_idx = idxs[int(train_percent * x_window.shape[0]):]

    scaler = MinMaxScaler()

    scale_values = np.concatenate((x_window[train_idx].reshape(-1), y_window[train_idx].reshape(-1)))
    scale_values = np.unique(scale_values)

    scaler.fit(scale_values)

    x_window[train_idx] = scaler.transform(x_window[train_idx].reshape(-1)).reshape(len(train_idx), x_window.shape[1], x_window.shape[2])
    y_window[train_idx] = scaler.transform(y_window[train_idx].reshape(-1)).reshape(len(train_idx), y_window.shape[1])

    x_window[test_idx] = scaler.transform(x_window[test_idx].reshape(-1)).reshape(len(test_idx), x_window.shape[1], x_window.shape[2])
    y_window[test_idx] = scaler.transform(y_window[test_idx].reshape(-1)).reshape(len(test_idx), y_window.shape[1])

    return x_window, y_window
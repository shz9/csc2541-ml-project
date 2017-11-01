import pandas as pd
import numpy as np

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
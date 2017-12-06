import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def load_co2(file_path):
    co2 = pd.read_csv(file_path, header=None, index_col=None)

    co2 = co2[0].values

    return co2


def load_erie(file_path):
    erie = pd.read_csv(file_path, header=0, names=["date", "level"], index_col=None)

    erie = erie["level"].values

    return erie


def load_solar(file_path):
    solar = pd.read_csv(file_path, header=None, index_col=None)

    solar = solar[1].values

    return solar


def get_data(filepath, fun):
    data = fun(filepath)

    return data

def create_window_data(data, window_size):
    temp = []

    for i in range(len(data) - window_size):
        temp.append(data[i: i + window_size])

    temp = np.array(temp)

    x = temp[:, :-1]
    y = temp[:, -1]

    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=1)

    return x, y


def difference(x, interval):
    differences = []

    for i in range(interval, len(x)):
        differences.append(x[i] - x[i - interval])

    return differences


def inverse_difference(history, diff, interval):
    return diff + history[-interval]


def scale(x_train, y_train, x_test, y_test):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    scale_values = np.concatenate((x_train.reshape(-1), y_train.reshape(-1)))
    scale_values = np.unique(scale_values)
    scaler.fit(scale_values.reshape(-1, 1))

    x_train = scaler.transform(x_train.reshape(-1, 1)).reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
    y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape[0], y_train.shape[1])

    x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
    y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape[0], y_test.shape[1])

    return scaler, x_train, y_train, x_test, y_test


def invert_scale(scaler, values):
    if values.ndim == 1:
        values = np.reshape(values, (-1, 1))

    inverted = scaler.inverse_transform(values)
    inverted = np.reshape(inverted, -1)

    return inverted


def create_direct_data(data, args):
    x, y = create_window_data(data, args["window_length"] + 1)

    split = int(args["train_percent"] * len(x))

    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    scaler, x_train, y_train, x_test, y_test = scale(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test, scaler


def create_detrended_data(data, args):
    data, trend = detrend(data, args)

    x, y = create_window_data(data, args.window_length + 1)
    split = int(args.train_percent * len(x))

    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    scaler, x_train, y_train, x_test, y_test = scale(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test, scaler, trend


def create_differenced_data(data, diff_interval, window_length, train_percent):
    differenced = difference(data, diff_interval)
    x, y = create_window_data(differenced, window_length + 1)

    split = int(train_percent * len(x))

    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    scaler, x_train, y_train, x_test, y_test = scale(x_train, y_train, x_test, y_test)

    return data, x_train, y_train, x_test, y_test, scaler


def detrend(data, args):
    idx = int(len(data) * args.train_percent) - (args.window_length - 1)

    model = LinearRegression()
    model.fit(np.arange(idx).reshape(-1, 1), data[:idx])

    trend = model.predict(np.arange(len(data)).reshape(-1, 1))

    data = data - trend

    return data, trend

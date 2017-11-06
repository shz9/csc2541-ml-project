import numpy as np


def generate_data(samples, seq_len, fun=np.sin):
    temp  = np.linspace(-10, 10, samples * seq_len)
    y_train = fun(temp)

    x_train_x = temp
    x_train_y = np.zeros(samples * seq_len)
    x_train_y[1:] = y_train[0: y_train.shape[0] - 1]
    x_train = np.zeros(shape=(samples, seq_len, 2))
    x_train[:, :, 0] = x_train_x.reshape((samples, seq_len))
    x_train[:, :, 1] = x_train_y.reshape((samples, seq_len))
    y_train = np.reshape(y_train, (samples, seq_len))

    temp = np.linspace(30, 50, samples * seq_len)
    y_test = fun(temp)

    x_test_x = temp
    x_test_y = np.zeros(samples * seq_len)
    x_test_y[1:] = y_test[0: y_test.shape[0] - 1]
    x_test = np.zeros(shape=(samples, seq_len, 2))
    x_test[:, :, 0] = x_test_x.reshape((samples, seq_len))
    x_test[:, :, 1] = x_test_y.reshape((samples, seq_len))
    y_test = np.reshape(y_test, (samples, seq_len))

    return x_train, y_train, x_test, y_test


def generate_data_many_to_one(samples, seq_len, fun=np.sin):
    temp = np.random.uniform(-10, 10, samples * seq_len)
    temp = np.sort(temp)
    y_train = []
    x_train_x = []
    x_train_y = []

    for i in range(len(temp) - seq_len):
        x_train_x.append(temp[i: i + seq_len - 1])
        x_train_y.append(fun(np.concatenate((temp[i: i + seq_len - 2], np.array([0])))))
        y_train.append(fun(temp[i + seq_len - 2]))

    x_train_x = np.array(x_train_x)
    x_train_y = np.array(x_train_y)
    x_train = np.zeros((x_train_x.shape[0], x_train_x.shape[1], 2))
    x_train[:, :, 0] = x_train_x
    x_train[:, :, 1] = x_train_y
    y_train = np.array(y_train)

    temp = np.random.uniform(30, 50, samples * seq_len)
    temp = np.sort(temp)
    y_test = []
    x_test_x = []
    x_test_y = []

    for i in range(len(temp) - seq_len):
        x_test_x.append(temp[i: i + seq_len - 1])
        x_test_y.append(fun(np.concatenate((temp[i: i + seq_len - 2], np.array([0])))))
        y_test.append(fun(temp[i + seq_len - 1]))

    x_test_x = np.array(x_test_x)
    x_test_y = np.array(x_test_y)
    x_test = np.zeros((x_test_x.shape[0], x_test_x.shape[1], 2))
    x_test[:, :, 0] = x_test_x
    x_test[:, :, 1] = x_test_y
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


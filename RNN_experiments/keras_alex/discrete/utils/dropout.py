from utils.plot import *
from utils.utils import *
from models import create_dropout_rnn, create_dropout_rnn_stateful
from keras.regularizers import l2
from keras.initializers import RandomNormal, RandomUniform
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from utils.data import create_differenced_data, inverse_difference, invert_scale
from utils.mc import *
import numpy as np
import tensorflow as tf


def train_stateless(x_train, y_train, args, params):
    model = create_dropout_rnn(args["seq_len"], args["seq_dim"], hidden_units=params["hidden_units"],
                               activation=params["activation"], dropout_dense=params["dropout_rate"],
                               dropout_input=params["dropout_rate"], dropout_lstm=params["dropout_rate"],
                               dropout_recurrent=params["dropout_rate"],
                               kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"])

    model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
              verbose=False)

    return model.get_config(), model.get_weights(), params


def train_stateful(x_train, y_train, args, params):
    model = create_dropout_rnn_stateful(args["seq_len"], args["seq_dim"], hidden_units=params["hidden_units"],
                               activation=params["activation"], dropout_dense=params["dropout_rate"],
                               dropout_input=params["dropout_rate"], dropout_lstm=params["dropout_rate"],
                               dropout_recurrent=params["dropout_rate"],
                               kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"])

    for i in range(params["epochs"]):
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, verbose=False)
        model.reset_states()

    return model.get_config(), model.get_weights(), params


def create_param_grid_stateless():
    epochs = [100]
    batch_size = [25]
    dropout_rate = [0.1, 0.4]
    hidden_units = [200]
    kernel_initializer = [RandomNormal(stddev=0.05)]
    kernel_regularizer = [l2(0.0001)]
    activation = ["relu"]

    param_grid = {"epochs": epochs, "batch_size": batch_size, "dropout_rate": dropout_rate, "hidden_units": hidden_units,
                  "kernel_initializer": kernel_initializer, "activation": activation,
                  "kernel_regularizer": kernel_regularizer}
    param_grid = ParameterGrid(param_grid)
    param_grid = list(param_grid)

    return param_grid


def create_param_grid_stateful():
    epochs = [50]
    batch_size = [25]
    dropout_rate = [0.1]
    hidden_units = [50, 100, 200]
    kernel_initializer = [RandomNormal(stddev=0.05)]
    kernel_regularizer = [l2(0.0001)]
    activation = ["tanh"]

    param_grid = {"epochs": epochs, "batch_size": batch_size, "dropout_rate": dropout_rate, "hidden_units": hidden_units,
                  "kernel_initializer": kernel_initializer, "activation": activation,
                  "kernel_regularizer": kernel_regularizer}
    param_grid = ParameterGrid(param_grid)
    param_grid = list(param_grid)

    return param_grid


def train_models_differences(data, args):
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args["diff_interval"], args["window_length"],
                                                                             args["train_percent"])
    if args["state"] == "stateless":
        train_fn = train_stateless
        param_grid = create_param_grid_stateless()
    else:
        train_fn = train_stateful
        param_grid = create_param_grid_stateful()

    if args["starts"] == "multiple":
        pred_fn = predict_different_start_points
        plot_fn = make_plots_differences_multiple
    else:
        pred_fn = mc_forward
        plot_fn = make_plots_differences_single

    for i in range(0, len(param_grid), 10):
        np.random.seed(100)
        tf.set_random_seed(100)

        temp_param = param_grid[i:i + 10]

        models = Parallel(n_jobs=10)(delayed(train_fn)(x_train, y_train, args, param) for param in temp_param)
        print("Done training models!!!")
        params = [temp[2] for temp in models]
        models = [(temp[0], temp[1]) for temp in models]

        predictions = Parallel(n_jobs=5)(delayed(pred_fn)(model[0], model[1], x_train, len(y_test), n_samples=10, state=args["state"]) for model in models)

        print("Done making predictions!!!")

        for j, y_pred in enumerate(predictions):
            num = j + (i * 10)
            plot_fn(data, x_train, y_train, x_test, y_test, scaler, y_pred, args, params[j], num)


def undo_differences_single(data, x_train, x_test, y_true, y_pred, args):
    final = []
    expected = []

    for i in range(len(x_train)):
        inverted_pred = inverse_difference(data, y_pred[:, i], len(x_train) + len(x_test) + args["diff_interval"] - i)
        inverted_true = inverse_difference(data, y_true[i], len(x_train) + len(x_test) + args["diff_interval"] - i)

        final.append(inverted_pred)
        expected.append(inverted_true)

    prev_history = [data[-(len(y_true) + args["diff_interval"] - len(x_train))]]

    for i in range(len(x_train), len(y_true)):
        inverted_pred = inverse_difference(prev_history, y_pred[:, i], 1)
        inverted_true = inverse_difference(data, y_true[i], len(y_true) + args["diff_interval"] - i)

        prev_history.append(inverted_pred)
        final.append(inverted_pred)
        expected.append(inverted_true)

    final = np.array(final)
    final = final.T
    expected = np.array(expected)

    return expected, final


def undo_differences_multiple(data, x_train, x_test, y_true, y_pred, args):
    final = []
    expected = []

    for i in range(len(x_train)):
        inverted_true = inverse_difference(data, y_true[i], len(x_train) + len(x_test) + args["diff_interval"] - i)
        expected.append(inverted_true)

    for i in range(len(x_train), y_pred.shape[2]):
        inverted_true = inverse_difference(data, y_true[i], len(y_true) + args["diff_interval"] - i)
        expected.append(inverted_true)

    for j in range(y_pred.shape[0]):
        final_temp = []

        for i in range(len(x_train) - j - 1):
            inverted_pred = inverse_difference(data, y_pred[j, :, i], len(x_train) + len(x_test) + args["diff_interval"] - i)
            final_temp.append(inverted_pred)

        final.append(final_temp)

    for j in range(y_pred.shape[0]):
        prev_history = [data[-(len(y_true) + args["diff_interval"] - len(x_train)) - j - 1]]

        for i in range(len(x_train) - j - 1, y_pred.shape[2]):
            inverted_pred = inverse_difference(prev_history, y_pred[j, :, i], 1)

            prev_history.append(inverted_pred)
            final[j].append(inverted_pred)

    final = np.array(final)
    final = np.transpose(final, (0, 2, 1))
    expected = np.array(expected)

    return expected, final


def make_plots_differences_single(data, x_train, y_train, x_test, y_test, scaler, y_pred, args, params, num):
    mean_pred = mc_mean(y_pred)
    make_residual_plot(y_train, y_test, mean_pred, args, params, num)

    y_pred = [invert_scale(scaler, y_pred[i]) for i in range(len(y_pred))]
    y_pred = np.array(y_pred)
    y_true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    expected, final = undo_differences_single(data, x_train, x_test, y_true, y_pred, args)

    mean_pred = mc_mean(final)
    std_pred = mc_std(final)

    make_prediction_plot(x_train, expected, mean_pred, args, params, num)
    make_error_plot(x_train, expected, mean_pred, std_pred, args, params, num)


def make_plots_differences_multiple(data, x_train, y_train, x_test, y_test, scaler, y_pred, args, params, num):
    unscaled_pred = []

    for i in range(y_pred.shape[0]):
        temp = []
        for j in range(y_pred.shape[1]):
            temp.append(invert_scale(scaler, y_pred[i, j]))

        unscaled_pred.append(temp)

    y_pred = np.array(unscaled_pred)
    y_true = invert_scale(scaler, np.concatenate((y_train, y_test)))

    expected, final = undo_differences_multiple(data, x_train, x_test, y_true, y_pred, args)

    mean_pred = np.mean(final, axis=1)
    std_pred = np.std(final, axis=1)

    make_prediction_plot(x_train, expected, mean_pred, args, params, num)

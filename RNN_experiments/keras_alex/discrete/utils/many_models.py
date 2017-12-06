from utils.plot import make_prediction_plot, make_residual_plot
from models import create_model_stateless, create_model_stateful
from keras.initializers import RandomNormal
from keras.regularizers import l1_l2, l2
from sklearn.model_selection import ParameterGrid
from utils.data import create_differenced_data, invert_scale, inverse_difference
import numpy as np
from joblib import Parallel, delayed
from utils.utils import predict_sliding


def train_stateless(x_train, y_train, args, params):
    model = create_model_stateless(args["seq_len"], args["seq_dim"], hidden_units=params["hidden_units"],
                               activation=params["activation"], hidden_layers=params["hidden_layers"],
                               kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"])

    model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
              verbose=False)

    return model.get_config(), model.get_weights(), params


def train_stateful(x_train, y_train, args, params):
    model = create_model_stateful(1, args["seq_len"], args["seq_dim"], hidden_units=params["hidden_units"],
                               activation=params["activation"], hidden_layers=params["hidden_layers"],
                               kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"])

    for i in range(params["epochs"]):
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, verbose=False)
        model.reset_states()

    return model.get_config(), model.get_weights(), params


def create_param_grid_stateless():
    epochs = [50, 100]
    batch_size = [25]
    hidden_units = [100, 150, 200]
    hidden_layers = [1, 2, 3]
    kernel_initializer = [RandomNormal(stddev=0.05)]
    kernel_regularizer = [l2(0.0001)]
    activation = ["relu"]

    param_grid = {"epochs": epochs, "batch_size": batch_size, "hidden_units": hidden_units, "hidden_layers": hidden_layers,
                  "kernel_initializer": kernel_initializer, "activation": activation,
                  "kernel_regularizer": kernel_regularizer}
    param_grid = ParameterGrid(param_grid)
    param_grid = list(param_grid)

    return param_grid


def create_param_grid_stateful():
    epochs = [50, 100, 200]
    batch_size = [1]
    hidden_units = [25, 50, 100]
    hidden_layers = [2, 3]
    kernel_initializer = [RandomNormal(stddev=0.05)]
    kernel_regularizer = [l2(0.0001)]
    activation = ["tanh"]

    param_grid = {"epochs": epochs, "batch_size": batch_size, "hidden_units": hidden_units, "hidden_layers": hidden_layers,
                  "kernel_initializer": kernel_initializer, "activation": activation,
                  "kernel_regularizer": kernel_regularizer}
    param_grid = ParameterGrid(param_grid)
    param_grid = list(param_grid)

    return param_grid


def train_models_differences(data, args):
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args["diff_interval"],
                                                                             args["window_length"],
                                                                             args["train_percent"])
    if args["state"] == "stateless":
        train_fn = train_stateless
        param_grid = create_param_grid_stateless()
    else:
        train_fn = train_stateful
        param_grid = create_param_grid_stateful()

    pred_fn = predict_sliding
    plot_fn = make_plots_differences

    for i in range(0, len(param_grid), 10):
        temp_param = param_grid[i:i + 10]

        models = Parallel(n_jobs=10)(delayed(train_fn)(x_train, y_train, args, param) for param in temp_param)
        print("Done training models!!!")
        params = [temp[2] for temp in models]
        models = [(temp[0], temp[1]) for temp in models]

        predictions = Parallel(n_jobs=10)(
            delayed(pred_fn)([model[0], model[1]], x_train, len(y_test), param["batch_size"], args["seq_len"]) for model, param in
            zip(models, params))

        print("Done making predictions!!!")

        for j, y_pred in enumerate(predictions):
            num = j + (i * 10)
            plot_fn(data, x_train, y_train, x_test, y_test, scaler, y_pred, args, params[j], num)


def make_plots_differences(data, x_train, y_train, x_test, y_test, scaler, y_pred, args, params, num):
    make_residual_plot(y_train, y_test, y_pred, args, params, num)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    expected, final = undo_differences_single(data, x_train, x_test, y_true, y_pred, args)

    make_prediction_plot(x_train, expected, final, args, params, num)


def undo_differences_single(data, x_train, x_test, y_true, y_pred, args):
    final = []
    expected = []

    for i in range(len(x_train)):
        inverted_pred = inverse_difference(data, y_pred[i], len(x_train) + len(x_test) + args["diff_interval"] - i)
        inverted_true = inverse_difference(data, y_true[i], len(x_train) + len(x_test) + args["diff_interval"] - i)

        final.append(inverted_pred)
        expected.append(inverted_true)

    prev_history = [data[-(len(y_true) + args["diff_interval"] - len(x_train))]]

    for i in range(len(x_train), len(y_true)):
        inverted_pred = inverse_difference(prev_history, y_pred[i], 1)
        inverted_true = inverse_difference(data, y_true[i], len(y_true) + args["diff_interval"] - i)

        prev_history.append(inverted_pred)
        final.append(inverted_pred)
        expected.append(inverted_true)

    final = np.array(final)
    expected = np.array(expected)

    return expected, final

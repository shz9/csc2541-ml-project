from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform, RandomNormal
from keras.models import save_model
from keras.callbacks import TensorBoard
from models import create_model_stateless
from utils.utils import *
from utils.data import *
from utils.plot import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=20, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.7, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--erie", default="data/monthly-lake-erie-levels-1921-19.csv", dest="erie")
parser.add_argument("--solar", default="data/solar_irradiance.csv", dest="solar")
parser.add_argument("--window-length", default=20, dest="window_length")
parser.add_argument("--epochs", default=50, dest="epochs")
parser.add_argument("--differences-predictions", default="results/stateless_differences_solar_predictions.png", dest="differences_predictions")
parser.add_argument("--differences_residuals", default="results/stateless_differences_solar_residuals.png", dest="differences_residuals")
parser.add_argument("--direct_predictions", default="results/stateless_direct_solar_predictions.png", dest="direct_predictions")
parser.add_argument("--direct_residuals", default="results/stateless_direct_solar_residuals.png", dest="direct_residuals")
parser.add_argument("--cv", default=False, dest="cv")
parser.add_argument("--diff-interval", default=1, dest="diff_interval")

BEST_PARAMS = {"epochs": 50, "kernel_initializer": RandomNormal(), "kernel_regularizer": l1_l2(0.0001, 0.0001),
               "hidden_units": 200, "activation": "tanh", "batch_size": 25,
               "hidden_layers": 1}


def create_grid_search(args, num_sequences):
    epochs = [1000]
    batch_size = [25]
    initializers = [RandomNormal()]
    regularizers = [l1_l2(0.0001, 0.0001)]
    hidden_units = [75]
    hidden_layers = [2]
    activation = ["relu"]

    model = KerasRegressor(build_fn=create_model_stateless, seq_len=args.seq_len, seq_dim=args.seq_dim, batch_size=BEST_PARAMS["batch_size"],
                           epochs=50)
    param_grid = {"kernel_initializer": initializers, "kernel_regularizer": regularizers, "epochs": epochs,
                  "hidden_units": hidden_units, "hidden_layers": hidden_layers,
                  "activation": activation, "batch_size": batch_size}


    idxs = np.arange(num_sequences)
    cv_splits = [(idxs[:int(args.train_percent * num_sequences)], idxs[int(args.train_percent * num_sequences):])]
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=10, scoring="neg_mean_squared_error", cv=cv_splits)

    return grid


def evaluate_grid_search(grid, x, y):
    results = grid.fit(x, y)

    best_model = results.best_estimator_
    print_grid_results(results)

    save_model(best_model.model, "best_stateless_model.h5")

    return best_model.model


def train_stateless(x_train, y_train, params, x_test=None, y_test=None, args=None):
    model = create_model_stateless(args.seq_len, args.seq_dim, kernel_initializer=params["kernel_initializer"],
                                   kernel_regularizer=params["kernel_regularizer"],
                                   hidden_units=params["hidden_units"], activation=params["activation"])

    if x_test is not None:
        model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                  validation_data=(x_test, y_test), shuffle=False)
    else:
        model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])

    return model


def train_differences(args):
    data = load_co2(args.co2)
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args.diff_interval, args.window_length,
                                                                             args.train_percent)

    if args.cv:
        grid = create_grid_search(args, x_train.shape[0])
        model = evaluate_grid_search(grid, x_train, y_train)
    else:
        model = train_stateless(x_train, y_train, BEST_PARAMS, args=args, x_test=x_test, y_test=y_test)

    y_pred = predict_sliding(model, x_train, x_test.shape[0], batch_size=BEST_PARAMS["batch_size"], seq_len=args.seq_len)

    # plot_residuals(np.arange(len(y_pred)), np.concatenate((y_train, y_test)), y_pred, len(x_train), "Stateless Difference Residuals", args.differences_residuals,
    #                BEST_PARAMS)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    final = []
    expected = []

    for i in range(len(x_train)):
        inverted_true = inverse_difference(data, y_true[i], len(x_train) + len(x_test) + args.diff_interval - i)
        inverted_pred = inverse_difference(data, y_pred[i], len(x_train) + len(x_test) + args["diff_interval"] - i)

        expected.append(inverted_true)
        final.append(inverted_pred)

    prev_history = [data[len(x_train) + 1]]

    for i in range(len(x_train), len(y_pred)):
        inverted_true = inverse_difference(data, y_true[i], len(y_true) + args.diff_interval - i)
        inverted_pred = inverse_difference(prev_history, y_pred[i], 1)

        expected.append(inverted_true)
        prev_history.append(inverted_pred)
        final.append(inverted_pred)

    final = np.array(final)
    expected = np.array(expected)

    plot_predictions(np.arange(len(expected)), expected, final, len(x_train), "Sliding Pred Stateless Differences",
                     args.differences_predictions, BEST_PARAMS)


def train_direct_observations(args):
    data = load_solar(args.solar)
    x_train, y_train, x_test, y_test, scaler, trend = create_detrended_data(data, args)

    if args.cv:
        grid = create_grid_search(args, x_train.shape[0])
        model = evaluate_grid_search(grid, x_train, y_train)
    else:
        model = train_stateless(x_train, y_train, BEST_PARAMS, args=args)

    y_pred = predict_sliding(model, x_train, x_test.shape[0], args)
    y_true = np.concatenate((y_train, y_test))

    plot_residuals(np.arange(len(y_true)), y_true, y_pred, len(x_train), "Stateless Direct Residuals", args.direct_residuals,
                   BEST_PARAMS)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, y_true)

    y_pred = y_pred + trend[args.window_length + 1:]
    y_true = y_true + trend[args.window_length + 1:]

    plot_predictions(np.arange(len(y_true)), y_true, y_pred, len(x_train), "Sliding Pred Stateless Direct",
                     args.direct_predictions, BEST_PARAMS)


def main(args):
    # train_direct_observations(args)
    train_differences(args)


if __name__ == "__main__":
    main(parser.parse_args())
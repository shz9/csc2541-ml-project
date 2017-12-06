from utils.utils import predict_sliding
from utils.data import *
from utils.plot import *
import argparse
import numpy as np
from highlevel.models import load_model
from highlevel.initializers import RandomNormal, RandomUniform
from highlevel.regularizers import l1_l2
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from train_stateless import train_stateless, BEST_PARAMS
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=20, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.7, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--erie", default="data/monthly-lake-erie-levels-1921-19.csv", dest="erie")
parser.add_argument("--window-length", default=20, dest="window_length")
parser.add_argument("--batch-size", default=50, dest="batch_size")
parser.add_argument("--epochs", default=50, dest="epochs")
parser.add_argument("--models", default="models/stateless/", dest="models")
parser.add_argument("--predictions", default="predictions/", dest="predictions")
parser.add_argument("--model-fn", default="stateless", dest="model_fn")
parser.add_argument("--diff-interval", default=1, dest="diff_interval")


def create_param_grid():
    epochs = [BEST_PARAMS["epochs"]]
    batch_size = [BEST_PARAMS["batch_size"]]
    dropout_rate = [0.01]
    hidden_units = [100, 150, 200]
    kernel_initializer = [RandomNormal(stddev=0.05), RandomNormal(stddev=0.1), RandomNormal(stddev=0.125),
                          RandomUniform(-0.05, 0.05), RandomUniform(-0.1, 0.1), RandomUniform(-0.125, 0.125)]
    kernel_regularizer = [l1_l2(0.01, 0.01)]
    activation = ["tanh", "sigmoid"]

    param_grid = {"epochs": epochs, "batch_size": batch_size, "dropout_rate": dropout_rate, "hidden_units": hidden_units,
                  "kernel_initializer": kernel_initializer, "activation": activation, "kernel_regularizer": kernel_regularizer}
    param_grid = ParameterGrid(param_grid)
    param_grid = list(param_grid)

    return param_grid


def train_many_models_differences(args):
    """
    This function trains many neural net models at once based on the command line arguments provided. Models
    are saved to file and reloaded (due to pickle issue), predictions for each model are then made and saved to file
    and reloaded, mean and std of predictions are calculated, and finally plots are made showing uncertainty in
    predictions.

    Args:
        args: Arguments parsed using argparse

    Returns:
        None
    """
    data = load_erie(args.erie)
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args)

    param_grid = create_param_grid()
    Parallel(n_jobs=10)(delayed(train_stateless)(x_train, y_train, param, i=i, args=args) for i, param in enumerate(param_grid))
    paths = [join(args.models, f) for f in listdir(args.models) if isfile(join(args.models, f))]

    Parallel(n_jobs=10)(delayed(parallel_predictions)(f, data, x_train, x_test, scaler, i, args) for i, f in enumerate(paths))
    paths = [join(args.predictions, f) for f in listdir(args.predictions) if isfile(join(args.predictions, f))]

    predictions = [pickle.load(open(path, "rb")) for path in paths]
    predictions = np.array(predictions)
    true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    expected = []

    for i in range(len(x_train)):
        inverted_true = inverse_difference(data, true[i], len(x_train) + len(x_test) + args.diff_interval - i)

        expected.append(inverted_true)


    for i in range(len(x_train), len(true)):
        inverted_true = inverse_difference(data, true[i], len(true) + args.diff_interval - i)

        expected.append(inverted_true)

    true = np.array(expected)
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

    plot_predictions(np.arange(len(true)), true, predictions, len(x_train), args.model_fn + "ensemble", args.model_fn + "_ensemble_hyp_erie_overlap.png", BEST_PARAMS)
    plot_error(np.arange(len(true)), true, mean, std, len(x_train), args.model_fn + "ensemble", args.model_fn + "_ensemble_hyp_erie_error.png")


def parallel_predictions(file, data, x_train, x_test, scaler, j, args):
    """
    This function takes a path to a model file and makes predictions for the test interval.

    Args:
        file: Path to a model file.
        x_train: Training data
    """
    model = load_model(file)

    pred = predict_sliding(model, x_train, x_test.shape[0], args)
    pred = invert_scale(scaler, pred)

    final = []

    for i in range(len(x_train)):
        inverted_pred = inverse_difference(data, pred[i], len(x_train) + len(x_test) + args.diff_interval - i)

        final.append(inverted_pred)

    prev_history = [data[len(x_train) + 1]]

    for i in range(len(x_train), len(pred)):
        inverted_pred = inverse_difference(prev_history, pred[i], 1)

        prev_history.append(inverted_pred)
        final.append(inverted_pred)

    final = np.array(final)

    pickle.dump(final, open("predictions/pred_" + str(j), "wb"))


def main(args):
    train_many_models_differences(args)


if __name__ == "__main__":
    main(parser.parse_args())

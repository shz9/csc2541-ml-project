from utils.utils import predict_sliding
from utils.data import *
from utils.plot import *
import argparse
import numpy as np
from models import create_model_stateless
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
import _pickle as pickle
from train_stateless import BEST_PARAMS
from models import reinstantiate_model


def train_stateless(x_train, y_train, params, args):
    model = create_model_stateless(args["seq_len"], args["seq_dim"], kernel_initializer=params["kernel_initializer"],
                                   kernel_regularizer=params["kernel_regularizer"],
                                   hidden_units=params["hidden_units"], activation=params["activation"],
                                   hidden_layers=params["hidden_layers"])

    model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)

    return model.get_config(), model.get_weights()


def train_many_models_direct(args):
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
    data = load_co2(args["co2"])
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args["diff_interval"],
                                                                              args["window_length"], args["train_percent"])

    models = Parallel(n_jobs=10)(delayed(train_stateless)(x_train, y_train, BEST_PARAMS, args=args) for i in range(50))
    predictions = Parallel(n_jobs=10)(delayed(parallel_predictions)(model[0], model[1], data, x_train, x_test, scaler, args, BEST_PARAMS) for model in models)
    predictions = np.array(predictions)

    true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    expected = []

    for i in range(len(x_train)):
        inverted_true = inverse_difference(data, true[i], len(x_train) + len(x_test) + args["diff_interval"] - i)
        expected.append(inverted_true)


    for i in range(len(x_train), len(true)):
        inverted_true = inverse_difference(data, true[i], len(true) + args["diff_interval"] - i)
        expected.append(inverted_true)

    true = np.array(expected)

    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

    plot_predictions_prod(np.arange(len(true)), true, predictions, len(x_train),
                     "CO2 Concentration (PPM)", "random_restarts_ensemble_nohyp_erie_overlap.png")
    plot_error_prod(np.arange(len(true)), true, mean, std, len(x_train),
               "CO2 Concentration (PPM)", "random_restarts_ensemble_nohyp_erie_error.png")


def parallel_predictions(config, weights, data, x_train, x_test, scaler, args, params):
    """
    This function takes a path to a model file and makes predictions for the test interval.

    Args:
        file: Path to a model file.
        x_train: Training data
    """
    model = reinstantiate_model(config, weights)

    pred = predict_sliding(model, x_train, x_test.shape[0], params["batch_size"], args["seq_len"])
    pred = invert_scale(scaler, pred)

    final = []

    for i in range(len(x_train)):
        inverted_pred = inverse_difference(data, pred[i], len(x_train) + len(x_test) + args["diff_interval"] - i)

        final.append(inverted_pred)

    prev_history = [data[len(x_train) + 1]]

    for i in range(len(x_train), len(pred)):
        inverted_pred = inverse_difference(prev_history, pred[i], 1)

        prev_history.append(inverted_pred)
        final.append(inverted_pred)

    final = np.array(final)

    return final

def main(args):
    train_many_models_direct(args)


if __name__ == "__main__":
    args = {"seq_len": 20, "seq_dim": 1, "train_percent": 0.7, "co2": "data/mauna-loa-atmospheric-co2.csv",
            "erie": "data/monthly-lake-erie-levels-1921-19.csv", "solar": "data/solar_irradiance.csv",
            "window_length": 20, "difference": "results/dropout/differences/",
            "direct": "results/dropout/direct/",
            "diff_interval": 1, "model_type": "stateless", "stationarity": "difference", "state": "stateless",
            "starts": "multiple"}

    main(args)
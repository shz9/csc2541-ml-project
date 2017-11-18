from utils.utils import predict_sliding
from utils.data import *
from utils.plot import *
import argparse
import numpy as np
from keras.models import load_model
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from train_stateless import train_stateless, BEST_PARAMS

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
    x_train, y_train, x_test, y_test, scaler, trend = create_direct_data_erie(args)

    Parallel(n_jobs=10)(delayed(train_stateless)(x_train, y_train, i=i, args=args) for i in range(10))
    paths = [join(args.models, f) for f in listdir(args.models) if isfile(join(args.models, f))]

    Parallel(n_jobs=10)(delayed(parallel_predictions)(f, x_train, x_test, scaler, trend, i, args) for i, f in enumerate(paths))
    paths = [join(args.predictions, f) for f in listdir(args.predictions) if isfile(join(args.predictions, f))]

    predictions = [pickle.load(open(path, "rb")) for path in paths]
    predictions = np.array(predictions)
    true = invert_scale(scaler, np.concatenate((y_train, y_test)))
    true = true + trend[args.window_length + 1:]

    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

    plot_predictions(np.arange(len(true)), true, predictions, args.model_fn + "ensemble", args.model_fn + "_ensemble_erie_overlap.png", BEST_PARAMS)
    plot_error(np.arange(len(true)), true, mean, std, args.model_fn + "ensemble", args.model_fn + "_ensemble_erie_error.png")


def parallel_predictions(file, x_train, x_test, scaler, trend, j, args):
    """
    This function takes a path to a model file and makes predictions for the test interval.

    Args:
        file: Path to a model file.
        x_train: Training data
    """
    model = load_model(file)

    pred = predict_sliding(model, x_train, x_test.shape[0], args)
    pred = invert_scale(scaler, pred)
    pred = pred + trend[args.window_length + 1:]
    pred = np.array(pred)

    pickle.dump(pred, open("predictions/pred_" + str(j), "wb"))


def main(args):
    train_many_models_direct(args)


if __name__ == "__main__":
    main(parser.parse_args())

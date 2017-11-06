from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Constant, Zeros, RandomUniform, RandomNormal
from keras.models import save_model, load_model
from models import create_model_stateless
from utils.utils import evaluate_model_static, evaluate_model_sliding_stateless, print_grid_results
from utils.data import *
from utils.plot import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=2, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.7, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--window-length", default=2, dest="window_length")
parser.add_argument("--batch-size", default=1, dest="batch_size")
parser.add_argument("--epochs", default=30, dest="epochs")
parser.add_argument("--static-predictions", default="results/stateless_static_predictions.png", dest="static_predictions")
parser.add_argument("--sliding-predictions", default="results/stateless_sliding_predictions.png", dest="sliding_predictions")


def cv_pipeline(num_samples, args, build_fn=create_model_stateless):
    epochs = [60]
    initializers = [RandomUniform()]
    regularizers = [l2(0.01)]
    hidden_units = [200]
    hidden_layers = [2]
    dropout = [True, False]
    dropout_rate = [0.05, 0.1, 0.2]

    model = KerasRegressor(build_fn=build_fn, seq_len=args.seq_len, seq_dim=args.seq_dim, batch_size=args.batch_size,
                           epochs=50)
    param_grid = {"kernel_initializer": initializers, "kernel_regularizer": regularizers, "epochs": epochs,
                  "hidden_units": hidden_units, "hidden_layers": hidden_layers,
                  "use_dropout": dropout, "dropout_rate": dropout_rate}

    idxs = np.arange(num_samples)
    cv_splits = [(idxs[:int(args.train_percent * num_samples)], idxs[int(args.train_percent * num_samples):])]
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, scoring="neg_mean_squared_error", cv=cv_splits)

    return grid


def train(x, y, args):
    grid = cv_pipeline(x.shape[0], args)
    grid_result = grid.fit(x, y)

    best_model = grid_result.best_estimator_
    save_model(best_model.model, "best_model.h5")

    print_grid_results(grid_result)

    return best_model.model


def train_no_cv(x, y, args):
    idxs = np.arange(x.shape[0])
    train_idx = idxs[:int(args.train_percent * x.shape[0])]
    test_idx = idxs[int(args.train_percent * x.shape[0]):]

    x_train, y_train = x[train_idx,], y[train_idx, ]
    x_test, y_test = x[test_idx, ], y[test_idx, ]
    model = create_model_stateless(args.seq_len, args.seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=l1_l2(0.01, 0.01),
                         )

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_test, y_test))

    return model


def main(args):
    data = load_co2(args.co2)

    co2_x, co2_y = create_window_data(data[0].values, args.window_length + 1)
    co2_x, co2_y = scale_data(co2_x, co2_y, args.train_percent)
    time_x, time_y = create_window_data(data[1].values, args.window_length + 1)

    model = train_no_cv(co2_x, co2_y, args)
    y_pred = evaluate_model_static(model, co2_x, co2_y, args)
    plot_predictions(co2_x, co2_y, y_pred, "Stateless Static Predictions", args.static_predictions)

    y_pred = evaluate_model_sliding_stateless(model, co2_x, co2_y)
    plot_predictions(co2_x, co2_y, y_pred, "Stateless Sliding Predictions", args.sliding_predictions)


if __name__ == "__main__":
    main(parser.parse_args())
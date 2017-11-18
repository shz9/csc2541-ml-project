from models import create_model_stateful
from utils.utils import print_grid_results, predict_sliding
from utils.data import *
from utils.plot import *
import argparse
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras.initializers import RandomNormal, RandomUniform
from keras.regularizers import l1_l2
from keras.models import save_model
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=20, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.7, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--erie", default="data/monthly-lake-erie-levels-1921-19.csv", dest="erie")
parser.add_argument("--window-length", default=20, dest="window_length")
parser.add_argument("--batch-size", default=1, dest="batch_size")
parser.add_argument("--epochs", default=1, dest="epochs")
parser.add_argument("--differences-predictions", default="results/stateful_differences_co2_predictions.png", dest="differences_predictions")
parser.add_argument("--differences-residuals", default="results/stateful_differences_co2_erie_residuals.png", dest="differences_residuals")
parser.add_argument("--models", default="models/", dest="models")
parser.add_argument("--predictions", default="predictions/", dest="predictions")
parser.add_argument("--cv", default=False, dest="cv")
parser.add_argument("--diff-interval", default=1, dest="diff_interval")
parser.add_argument("--direct_predictions", default="results/stateful_direct_erie_co2_predictions.png", dest="direct_predictions")
parser.add_argument("--direct_residuals", default="results/stateful_direct_erie_co2_residuals.png", dest="direct_residuals")

BEST_PARAMS = {"epochs": 30, "kernel_initializer": RandomNormal(), "kernel_regularizer": l1_l2(0.001, 0.001),
               "hidden_units": 10, "activation": "tanh"}

def train_stateful(x_train, y_train, i=-1, x_test=None, y_test=None, args=None):
    model = create_model_stateful(args.batch_size, args.seq_len, args.seq_dim,
                                  activation=BEST_PARAMS["activation"],
                                  kernel_initializer=BEST_PARAMS["kernel_initializer"],
                                  kernel_regularizer=BEST_PARAMS["kernel_regularizer"],
                                  hidden_units=BEST_PARAMS["hidden_units"])

    for epoch in range(BEST_PARAMS["epochs"]):
        if x_test is not None:
            model.fit(x_train, y_train, epochs=1, batch_size=args.batch_size, shuffle=False, validation_data=(x_test, y_test))
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=args.batch_size, shuffle=False)

        model.reset_states()

    if i < 0:
        return model
    else:
        save_model(model, "models/model_" + str(i))


def create_grid_search(args, num_sequences):
    hidden_units = [5, 10, 15]
    epochs = [10]
    activation = ["tanh", "sigmoid"]
    hidden_layers = [1]
    use_dropout = [False]
    kernel_regularizer = [None, l1_l2(0.01, 0.01)]
    kernel_initializer = [RandomUniform(), RandomNormal()]
    dropout_rate = [0.0, 0.01, 0.05]

    param_grid = {"hidden_units": hidden_units,
                  "hidden_layers": hidden_layers, "use_dropout": use_dropout,
                  "kernel_regularizer": kernel_regularizer, "kernel_initializer": kernel_initializer,
                  "epochs": epochs, "dropout_rate": dropout_rate, "activation": activation}

    model = KerasRegressor(build_fn=create_model_stateful, batch_size=args.batch_size, seq_len=args.seq_len,
                           seq_dim=args.seq_dim, shuffle=False)

    idxes = np.arange(num_sequences)
    train_idxes = idxes[:int(args.train_percent * num_sequences)]
    test_idxes = idxes[int(args.train_percent * num_sequences):]

    cv = [(train_idxes, test_idxes)]
    grid = GridSearchCV(model, param_grid=param_grid, n_jobs=10, scoring="neg_mean_squared_error", cv=cv)

    return grid


def evaluate_grid_search(grid, x, y):
    results = grid.fit(x, y)

    best_model = results.best_estimator_
    save_model(best_model.model, "best_stateful_model.h5")

    print_grid_results(results)

    return best_model.model


def train_differences(args):
    data = load_co2(args.co2)
    data, x_train, y_train, x_test, y_test, scaler = create_differenced_data(data, args)

    if args.cv:
        grid = create_grid_search(args, x_train.shape[0])
        model = evaluate_grid_search(grid, x_train, y_train)
    else:
        model = train_stateful(x_train, y_train, args=args)

    y_pred = predict_sliding(model, x_train, x_test.shape[0], args)
    y_true = np.concatenate((y_train, y_test))

    plot_residuals(np.arange(len(y_true)), y_true, y_pred, "Stateful Differences Residuals", args.differences_residuals,
                   BEST_PARAMS)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, y_true)
    final = []
    expected = []

    for i in range(len(x_train)):
        inverted_pred = inverse_difference(data, y_pred[i], len(x_train) + len(x_test) + args.diff_interval - i)
        inverted_true = inverse_difference(data, y_true[i], len(x_train) + len(x_test) + args.diff_interval - i)

        final.append(inverted_pred)
        expected.append(inverted_true)

    prev_history = [data[len(y_train) + 1]]

    for i in range(len(x_train), len(y_pred)):
        inverted_pred = inverse_difference(prev_history, y_pred[i], 1)
        inverted_true = inverse_difference(data, y_true[i], len(y_true) + args.diff_interval - i)

        prev_history.append(inverted_pred)
        final.append(inverted_pred)
        expected.append(inverted_true)

    final = np.array(final)
    expected = np.array(expected)

    plot_predictions(np.arange(len(final)), expected, final, "Sliding Pred Stateful Differences", args.differences_predictions,
                     BEST_PARAMS)


def train_direct_observations(args):
    data = load_co2(args.co2)
    x_train, y_train, x_test, y_test, scaler, trend = create_direct_data(data, args)

    if args.cv:
        grid = create_grid_search(args, x_train.shape[0])
        model = evaluate_grid_search(grid, x_train, y_train)
    else:
        model = train_stateful(x_train, y_train, args=args)

    y_pred = predict_sliding(model, x_train, x_test.shape[0], args)
    y_true = np.concatenate((y_train, y_test))

    plot_residuals(np.arange(len(y_true)), y_true, y_pred, "Stateful Direct Residuals", args.direct_residuals,
                   BEST_PARAMS)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, y_true)

    y_pred = y_pred + trend[args.window_length + 1:]
    y_true = y_true + trend[args.window_length + 1:]

    plot_predictions(np.arange(len(y_true)), y_true, y_pred, "Sliding Pred Stateful Direct",
                     args.direct_predictions, BEST_PARAMS)


def main(args):
    train_differences(args)
    train_direct_observations(args)


if __name__ == "__main__":
    main(parser.parse_args())
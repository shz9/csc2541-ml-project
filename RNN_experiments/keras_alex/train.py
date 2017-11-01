from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Constant, Zeros, RandomUniform, RandomNormal
from keras.models import save_model, load_model
from model import create_model
from utils.utils import print_grid_results
from utils.data import *
from utils.plot import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=30, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.8, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--window-length", default=30, dest="window_length")


def cv_pipeline(num_samples, args):
    epochs = [50]
    initializers = [RandomUniform()]
    regularizers = [l2(0.01), l1_l2(0.1, 0.1), l1_l2(0.01, 0.01), l2(0.1)]
    hidden_units = [200]
    hidden_layers = [2]

    model = KerasRegressor(build_fn=create_model, seq_len=args.seq_len, seq_dim=args.seq_dim, batch_size=50, epochs=50)
    param_grid = {"kernel_initializer": initializers, "kernel_regularizer": regularizers, "epochs": epochs,
                  "hidden_units": hidden_units, "hidden_layers": hidden_layers}

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
    model = create_model(args.seq_len, args.seq_dim)

    model.fit(x, y, epochs=100, batch_size=50)

    return model


def evaluate_model_sliding(model, x, y):
    cur_window = x[0]
    pred = []

    for i in range(x.shape[0]):
        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0,0])
        cur_window = cur_window[1:]
        cur_window = np.concatenate((cur_window, np.array([[pred[i]]])))

    pred = np.array(pred)

    return pred


def evaluate_model_static(model, x, y):
    score = model.evaluate(x, y)

    print('\n Score: ')
    print(score)

    pred = model.predict(x)

    return pred


def main(args):
    data = load_co2(args.co2)

    co2_x, co2_y = create_window_data(data[0].values, args.window_length + 1)
    time_x, time_y = create_window_data(data[1].values, args.window_length + 1)

    model = train(co2_x, co2_y, args)
    y_pred = evaluate_model_static(model, co2_x, co2_y)
    plot_predictions(co2_x, co2_y, y_pred)

    y_pred = evaluate_model_sliding(model, co2_x, co2_y)
    plot_predictions(co2_x, co2_y, y_pred)


if __name__ == "__main__":
    main(parser.parse_args())
from utils.data import load_co2, invert_scale, create_direct_data
from utils.utils import predict_sliding
from models import create_model_stateful
from keras.initializers import RandomNormal
from keras.regularizers import l1_l2
from utils.plot import plot_residuals, plot_predictions
import numpy as np


BEST_PARAMS = {"epochs": 50, "kernel_initializer": RandomNormal(), "kernel_regularizer": l1_l2(0.0001, 0.0001),
               "hidden_units": 50, "activation": "tanh", "batch_size": 1, "hidden_layers": 2}


def fit_model(x_train, y_train, args, params, x_test=None, y_test=None):
    model = create_model_stateful(params["batch_size"], args["seq_len"], args["seq_dim"],
                                  activation=params["activation"],
                                  kernel_initializer=params["kernel_initializer"],
                                  kernel_regularizer=params["kernel_regularizer"],
                                  hidden_units=params["hidden_units"],
                                  hidden_layers=params["hidden_layers"])

    for epoch in range(BEST_PARAMS["epochs"]):
        if x_test is not None:
            model.fit(x_train, y_train, epochs=1, batch_size=params["batch_size"], shuffle=False, validation_data=(x_test, y_test))
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=params["batch_size"], shuffle=False)

        model.reset_states()

    return model


def train_direct(args, params):
    data = load_co2(args["co2"])

    x_train, y_train, x_test, y_test, scaler = create_direct_data(data, args)

    model = fit_model(x_train, y_train, args, params, x_test, y_test)

    y_pred = predict_sliding(model, x_train, x_test.shape[0], params["batch_size"], args["seq_len"])
    y_true = np.concatenate((y_train, y_test))

    plot_residuals(np.arange(len(y_true)), y_true, y_pred, len(x_train), "Stateless Direct Residuals", args["direct_residuals"],
                   params)

    y_pred = invert_scale(scaler, y_pred)
    y_true = invert_scale(scaler, y_true)

    plot_predictions(np.arange(len(y_true)), y_true, y_pred, len(x_train), "Sliding Pred Stateless Direct",
                     args["direct_predictions"], params)


def main():
    args = {"seq_len": 20, "seq_dim": 1, "train_percent": 0.7, "co2": "data/mauna-loa-atmospheric-co2.csv",
            "erie": "data/monthly-lake-erie-levels-1921-19.csv", "solar": "data/solar_irradiance.csv",
            "window_length": 20,
            "direct_predictions": "results/no_dropout/direct/temp_pred.png",
            "direct_residuals": "results/no_dropout/direct/temp_res.png",
            "diff_interval": 1, "model_type": "stateless", "stationarity": "difference", "state": "stateless",
            "starts": "multiple"}

    train_direct(args, BEST_PARAMS)


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(x, true, pred, idx, title, file_path, params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax1.plot(np.arange(x.shape[0]), pred.T, color="r", label="Predicted", alpha=0.2)
    ax1.set_title(title)
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax1.axvline(x=idx, color="black", linestyle="--")

    param_string = create_param_string(params)
    ax2.set_title("Params")
    ax2.text(0.1, 0.5, param_string, horizontalalignment="left", verticalalignment="center",
            transform=ax2.transAxes, wrap=True)

    fig.savefig(file_path, dpi=300)


def plot_error(x, true, mean, std, idx, title, file_path, params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax1.plot(np.arange(x.shape[0]), mean.reshape(-1), color="r", label="Pred")
    ax1.axvline(x=idx, color="black", linestyle="--")
    ax1.fill_between(np.arange(x.shape[0]), (mean + (2 * std)).reshape(-1), (mean - ( 2* std)).reshape(-1), color="r", alpha=0.2)
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax1.set_title(title)

    param_string = create_param_string(params)
    ax2.set_title("Params")
    ax2.text(0.1, 0.5, param_string, horizontalalignment="left", verticalalignment="center", transform=ax2.transAxes,
             wrap=True)

    fig.savefig(file_path, dpi=300)


def plot_residuals(x, true, pred, idx, title, file_path, params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.scatter(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax1.scatter(np.arange(x.shape[0]), pred.reshape(-1), color="r",  label="Pred")
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax1.set_title(title)
    ax1.axvline(x=idx, color="black", linestyle="--")

    param_string = create_param_string(params)
    ax2.set_title("Params")
    ax2.text(0.1, 0.5, param_string, horizontalalignment="left", verticalalignment="center", transform=ax2.transAxes,
             wrap=True)

    fig.savefig(file_path, dpi=300)


def plot_error_prod(x, true, mean, std, idx, ylab, file_path):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax.plot(np.arange(x.shape[0]), mean.reshape(-1), color="r", label="Pred")
    ax.axvline(x=idx, color="black", linestyle="--")
    ax.fill_between(np.arange(x.shape[0]), (mean + (2 * std)).reshape(-1), (mean - ( 2* std)).reshape(-1),
                    color="gray", alpha=0.2)
    ax.set_xlabel("Months")
    ax.set_ylabel(ylab)

    fig.savefig(file_path, dpi=600)


def plot_predictions_prod(x, true, pred, idx, ylab, file_path):
    fig = plt.figure()
    
    ax = fig.add_subplot(111)

    ax.plot(np.arange(x.shape[0]), true.reshape(-1), color="b", label="True")
    ax.plot(np.arange(x.shape[0]), pred.T, color="r", label="Predicted")
    ax.axvline(x=idx, color="black", linestyle="--")
    ax.set_xlabel("Months")
    ax.set_ylabel(ylab)

    fig.savefig(file_path, dpi=600)


def create_param_string(params):
    param_string = ""
    primitive = (int, str, bool, float)

    for key, value in params.items():
        if not isinstance(value, primitive) and value is not None:
            value = str(type(value)) + str(value.get_config())

        param_string += key + ": " + str(value)
        param_string += "\n"

    return param_string


def create_title_string(data_name, state, stationarity, plot_type):
    title = "{} {} {} {}".format(data_name, state, stationarity, plot_type)

    return title


def create_plot_file_name(dir, data_name, state, stationarity, starts, activation, epochs, drop, seq_len, train_percent, plot_type, num):
    path = "{}{}_{}_{}_{}_{}_{}_drop_{}_seq_{}_percent_{}_{}_{}.pdf".format(dir, data_name, state, stationarity, starts, activation,
                                                           epochs, drop,
                                                           seq_len, train_percent, plot_type, num)

    return path


def make_residual_plot(y_train, y_test, mean_pred, args, params, num, type="single_start"):
    file_path = create_plot_file_name(args[args["stationarity"]], args["data_name"], args["state"],
                                         args["stationarity"], args["starts"], params["activation"], params["epochs"],
                                      params.get("dropout_rate", "none"), args["seq_len"], args["train_percent"],
                                         "resid", num)

    plot_residuals(np.arange(len(np.concatenate((y_train, y_test)))), np.concatenate((y_train, y_test)), mean_pred, len(y_train),
                   create_title_string(args["data_name"], args["state"], args["stationarity"], "residuals"),
                   file_path, params)


def make_prediction_plot(x_train, expected, mean_pred, args, params, num):
    file_path = create_plot_file_name(args[args["stationarity"]], args["data_name"], args["state"],
                                         args["stationarity"], args["starts"], params["activation"], params["epochs"],
                                  params.get("dropout_rate", "none"), args["seq_len"], args["train_percent"],
                                         "preds", num)

    plot_predictions(np.arange(len(expected)), expected, mean_pred, len(x_train),
                     create_title_string(args["data_name"], args["state"], args["stationarity"], "predictions"),
                     file_path, params)


def make_error_plot(x_train, expected, mean_pred, std_pred, args, params, num):
    file_path = create_plot_file_name(args[args["stationarity"]], args["data_name"], args["state"],
                                         args["stationarity"], args["starts"], params["activation"], params["epochs"],
                                      params.get("dropout_rate", "none"), args["seq_len"], args["train_percent"],
                                         "conf int", num)

    # plot_error(np.arange(len(expected)), expected, mean_pred, std_pred, len(x_train),
    #            create_title_string(args["data_name"], args["state"], args["stationarity"], "confidence intervals"),
    #            file_path, params)
    plot_error_prod(np.arange(len(expected)), expected, mean_pred, std_pred, len(x_train),
               "CO2 Concentration (PPM)", file_path)
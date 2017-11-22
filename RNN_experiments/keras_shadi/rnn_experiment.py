import glob
import pickle
import numpy as np
import os
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model, save_model
from matplotlib import pyplot as plt
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from RNN_experiments.keras_shadi.bootstrap.other_bootstrap import stationary_boostrap_method, \
    moving_block_bootstrap_method, circular_block_bootstrap_method
from joblib import Parallel, delayed


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    training_acc = None
    for i in range(nb_epoch):
        train_res = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        training_acc = train_res.history['loss']
        model.reset_states()
    return model, training_acc


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def train_model(ith_dataset, idx):
    diff_values = ith_dataset['CO2Concentration'].diff()

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)

    lstm_model, loss = fit_lstm(supervised.values, 1, 100, 50)

    with open("./model_performance/" + str(idx) + ".pkl", "wb") as lf:
        pickle.dump(loss[0], lf)

    save_model(lstm_model, "./models/" + str(idx) + ".kmodel")


def make_model_predictions(train, all_y, init_val, idx):

    model = load_model("./models/" + str(idx) + ".kmodel")
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train[:, 0].reshape(len(train), 1, 1)

    model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = []
    prev = None
    prev_history = [init_val]  # initial
    for i in range(len(all_y)):
        # make one-step forecast
        if prev is None:
            prev = all_y[i, 0:-1]

        yhat = forecast_lstm(model, 1, prev)
        prev = yhat
        # reshape
        prev = np.array([prev])
        # invert scaling
        # yhat = invert_scale(scaler, X, yhat)
        # print('yhat after inverse scale: ', yhat)
        # invert differencing
        # yhat = inverse_difference(prev_history, yhat, i)
        yhat = yhat + prev_history[i]
        prev_history.append(yhat)
        # store forecast
        predictions.append(yhat)

    with open("./predictions/" + str(idx) + ".pkl", "wb") as pf:
        pickle.dump(predictions, pf)


def main(retrain=True, bootstrap_method='gp', n_rnns=10, plot_preds=True, filter_bad_models=True):

    # load dataset
    ex_dataset = pd.read_csv('../../data/mauna-loa-atmospheric-co2.csv',
                             header=None)
    ex_dataset.columns = ['CO2Concentration', 'Time']

    train_data = ex_dataset.loc[ex_dataset.Time <= 1980, ['CO2Concentration', 'Time']]

    if retrain:

        # Delete old models:
        for mod_path in glob.glob("./models/*.kmodel"):
            os.remove(mod_path)

        for mod_path in glob.glob("./model_performance/*.pkl"):
            os.remove(mod_path)

        if bootstrap_method == 'gp':

            # Due to issues with multiprocessing in sklearn not terminating
            # and training jobs for RNNs not starting as a results,
            # I had to generate GP samples in a separate script.
            # Will find a way to fix this issue later on.
            bootstrapped_dataset = []

            for gpf in glob.glob("./bootstrap/gp_samples/*.pkl"):
                with open(gpf, "rb") as gpfp:
                    bootstrapped_dataset.append(pickle.load(gpfp))

                if len(bootstrapped_dataset) == n_rnns:
                    break

        elif bootstrap_method == 'stationary_block':
            bootstrapped_dataset = stationary_boostrap_method(train_data['Time'],
                                                              train_data['CO2Concentration'],
                                                              n_samples=n_rnns)
        elif bootstrap_method == 'moving_block':
            bootstrapped_dataset = moving_block_bootstrap_method(train_data['Time'],
                                                                 train_data['CO2Concentration'],
                                                                 n_samples=n_rnns)
        elif bootstrap_method == 'circular_block':
            bootstrapped_dataset = circular_block_bootstrap_method(train_data['Time'],
                                                                   train_data['CO2Concentration'],
                                                                   n_samples=n_rnns)
        else:
            raise Exception("Bootstrap method not implemented!")

        print "Training ensemble..."
        Parallel(n_jobs=10, verbose=10)(delayed(train_model)(pd.DataFrame({'Time': np.ravel(dat[0]),
                                                                          'CO2Concentration': np.ravel(dat[1])}),
                                                             idx)
                                        for idx, dat in enumerate(bootstrapped_dataset))
        print "Done training ensemble!"

    for pred_path in glob.glob("./predictions/*.pkl"):
        os.remove(pred_path)

    diff_values = ex_dataset['CO2Concentration'].diff()
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-228], supervised_values[-228:]

    mod_pf = {}
    for modpf in glob.glob("./model_performance/*.pkl"):
        with open(modpf, "rb") as modof:
            mod_pf[int(os.path.basename(modpf).replace(".pkl", ""))] = pickle.load(modof)

    best_models = []

    for midx in mod_pf.keys():
        if np.sqrt(mod_pf[midx]) < .15:
            print np.sqrt(mod_pf[midx])
            best_models.append(midx)
        elif not filter_bad_models:
            best_models.append(midx)

    if len(best_models) < 1:
        raise Exception("None of the models achieved the minimum threshold for the MSE metric.")

    Parallel(n_jobs=20, verbose=10)(delayed(make_model_predictions)(train,
                                                                    supervised_values, # test,
                                                                    ex_dataset['CO2Concentration'][0], # ex_dataset['CO2Concentration'][len(test)+1],
                                                                    idx)
                                    for idx in best_models)

    preds = []

    for predf in glob.glob("./predictions/*.pkl"):
        with open(predf, "rb") as pf:
            preds.append(pickle.load(pf))

    print len(preds)

    #for mod in rnn_models:
    #    preds.append(make_model_predictions(mod, train, test, ex_dataset['CO2Concentration'][len(test)+1]))

    # line plot of observed vs predicted
    rnn_means = np.array([])
    rnn_conf = np.array([])

    for k in range(len(preds[0])):
        step_vals = [el[k] for el in preds]
        rnn_means = np.append(rnn_means, np.mean(step_vals))
        #rnn_means = np.append(rnn_means, stats.trim_mean(step_vals, 0.3))
        rnn_conf = np.append(rnn_conf, np.std(step_vals))

    #plt.plot(ex_dataset['CO2Concentration'][-228:])
    plt.plot(ex_dataset['CO2Concentration'], color="blue")
    if plot_preds:
        for pred in preds:
            plt.plot(pred, color="green", alpha=.1)
    plt.plot(rnn_means, color="red")
    plt.fill_between(list(range(len(rnn_means))),
                     rnn_means - rnn_conf,
                     rnn_means + rnn_conf,
                     color="gray", alpha=0.2)

    plt.axvline(len(train_data), color="purple", linestyle='--')

    plt.title("Bootstrap Method: " + bootstrap_method + " / Number of RNNs: " + str(len(best_models)))

    plt.savefig("./figures/" + bootstrap_method + "_trained" + str(n_rnns) + "_selected" + str(len(best_models)) +
                "_" + ["no_preds", "preds"][plot_preds] + ".png")


if __name__ == "__main__":
    main(retrain=True, bootstrap_method='gp',
         n_rnns=100, plot_preds=True, filter_bad_models=True)

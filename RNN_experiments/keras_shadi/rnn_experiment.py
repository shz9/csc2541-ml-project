import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model, save_model
from matplotlib import pyplot as plt
import numpy as np
import glob
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from RNN_experiments.bootstrap.gp_bootstrap import bootstrap_data


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
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print("Training Epoch", i, "done...")
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def train_model(ith_dataset, idx):
    diff_values = ith_dataset['CO2Concentration'].diff()

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)

    lstm_model = fit_lstm(supervised.values, 1, 50, 4)

    save_model(lstm_model, "./models/" + str(idx) + ".kmodel")


def make_model_predictions(model, train, all_y, init_val):

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

    return predictions


def main(retrain=True):

    # load dataset
    ex_dataset = pd.read_csv('../../data/mauna-loa-atmospheric-co2.csv',
                             header=None)
    ex_dataset.columns = ['CO2Concentration', 'Time']

    train_data = ex_dataset.loc[ex_dataset.Time <= 1980, ['CO2Concentration', 'Time']]

    if retrain:
        bootstrapped_dataset = bootstrap_data(train_data['Time'].reshape(-1, 1),
                                              train_data['CO2Concentration'].reshape(-1, 1),
                                              34.4**2 * RBF(length_scale=41.8) +
                                              3.27**2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                                                               periodicity=1) +
                                              0.446**2 * RationalQuadratic(alpha=17.7, length_scale=0.957) +
                                              0.197**2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336),
                                              samples=10)
        # Need to run this in parallel:

        # t_pool = ThreadPool(20)
        Parallel(n_jobs=10)(delayed(train_model)(pd.DataFrame({'Time': np.ravel(dat[0]),
                                                               'CO2Concentration': np.ravel(dat[1])}),
                                                 idx)
                            for idx, dat in enumerate(bootstrapped_dataset))

    """for x, y in bootstrapped_dataset:
        temp_df = pd.DataFrame({'Time': np.ravel(x),
                                'CO2Concentration': np.ravel(y)})
        rnn_models.append(train_model(temp_df))"""

    rnn_models = []
    for mod_path in glob.glob("./models/*.kmodel"):
        rnn_models.append(load_model(mod_path))

    diff_values = ex_dataset['CO2Concentration'].diff()
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-228], supervised_values[-228:]

    preds = []
    for mod in rnn_models:
        preds.append(make_model_predictions(mod, train, test, ex_dataset['CO2Concentration'][len(test)+1]))

    # line plot of observed vs predicted
    rnn_means = np.array([])
    rnn_conf = np.array([])

    for k in range(len(preds[0])):
        step_vals = [el[k] for el in preds]
        rnn_means = np.append(rnn_means, np.mean(step_vals))
        rnn_conf = np.append(rnn_conf, np.std(step_vals))

    plt.plot(ex_dataset['CO2Concentration'][-228:])
    plt.plot(rnn_means)
    plt.fill_between(list(range(len(rnn_means))),
                     rnn_means - rnn_conf,
                     rnn_means + rnn_conf,
                     color="gray", alpha=0.2)

    plt.show()


if __name__ == "__main__":
    main(False)

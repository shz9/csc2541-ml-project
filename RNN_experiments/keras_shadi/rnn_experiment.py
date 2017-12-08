"""
Author: Shadi Zabad
Date: November 2017
"""

import glob
import pickle
import numpy as np
import os
import pandas as pd
from tensorflow import set_random_seed
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model, save_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from data.data_reader import read_lake_erie_data, read_mauna_loa_co2_data
from RNN_experiments.keras_shadi.bootstrap.other_bootstrap import stationary_boostrap_method, \
    moving_block_bootstrap_method, circular_block_bootstrap_method
from joblib import Parallel, delayed

# ------------------------------------------------------------

DATASET = "co2"
DATASET_TITLE = "CO2 Dataset"
RETRAIN = True
BOOTSTRAP_METHOD = 'gp'
ENSEMBLE_SIZE = 20
BEST_PERFORMING_FRACTION = .8
N_JOBS = 10
N_TRAINING_EPOCHS = 100
GP_BOOTSTRAP_SAMPLES_PATH = "./bootstrap/gp_samples/co2/*.pkl"
SET_STATE_PERC = 0.7

X_LABEL = "Month"
Y_LABEL = "CO2 Concentration (PPM)"

# ------------------------------------------------------------

# Set the seed
np.random.seed(1)
set_random_seed(2)

# ------------------------------------------------------------


def timeseries_to_supervised(data, lag=1):

    # Transform the data to differences between y[t+1] and y[t]:
    if isinstance(data[1], np.ndarray):
        t_data = pd.Series(data[1])
    else:
        t_data = data[1]

    data = t_data.diff()

    df = pd.DataFrame(data)

    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)

    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)

    X, y = df.values[:, 0:-1], df.values[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, y


def create_lstm_model(input_shape=(1, 1, 1), neurons=50, n_lstm_layers=3):

    if n_lstm_layers < 1:
        raise Exception("The model should have at least 1 lstm layer.")

    # Create and compile the model in Keras:
    model = Sequential()

    # Modified the code so that all models are initialized with zeros.
    # This should isolate uncertainty due to initialization from
    # uncertainty in the data.
    if n_lstm_layers == 1:
        model.add(LSTM(neurons,
                       batch_input_shape=input_shape,
                       stateful=True))
    else:
        model.add(LSTM(neurons,
                       batch_input_shape=input_shape,
                       stateful=True,
                       return_sequences=True
                       ))

        for nl in range(n_lstm_layers - 2):
            model.add(LSTM(neurons,
                           stateful=True,
                           return_sequences=True
                           ))

        model.add(LSTM(neurons,
                       stateful=True
                       ))

    # Add final layer:
    model.add(Dense(1))

    # Compile model:
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train_model(Xi, Yi, idx):

    # Create the LSTM model
    lstm_model = create_lstm_model(input_shape=(1, Xi.shape[1], Xi.shape[2]))

    # Train the model for <n_epochs>
    for i in range(N_TRAINING_EPOCHS):
        lstm_model.fit(Xi, Yi, epochs=1, batch_size=1, verbose=0, shuffle=False)
        lstm_model.reset_states()

    # The following 3 steps are meant to check how the model is doing on the training data
    # This may be used by later steps for filtering bad models from ensemble
    y_train_pred = lstm_model.predict(Xi, batch_size=1)

    lstm_model.reset_states()

    mse = mean_squared_error(Yi, y_train_pred)
    print idx, mse

    # Save performance score and trained model:

    with open("./model_performance/" + DATASET + "/" + str(idx) + ".pkl", "wb") as lf:
        pickle.dump(mse, lf)

    save_model(lstm_model, "./models/" + DATASET + "/" + str(idx) + ".kmodel")


def lstm_predict(ground_truth, model_idx):

    model = load_model("./models/" + DATASET + "/" + str(model_idx) + ".kmodel")
    predictions = [ground_truth[0]]

    if SET_STATE_PERC == 0:
        prev = np.array([ground_truth[1] - ground_truth[0]])
    else:
        prev = np.zeros(1)

    for i in range(len(ground_truth)):

        if i < SET_STATE_PERC * len(ground_truth):
            # Use these measurements to set the state of the model:
            yhat = ground_truth[i + 1] - ground_truth[i]
        else:
            yhat = model.predict(prev.reshape(1, 1, -1), batch_size=1)[0, 0]

        prev = np.array([yhat])
        predictions.append(yhat + predictions[i])

    with open("./predictions/" + str(model_idx) + ".pkl", "wb") as pf:
        pickle.dump(predictions, pf)


def bootstrap_data(train_data, bootstrap_method, n_samples):

    if bootstrap_method == 'gp':

        # Due to issues with multiprocessing in sklearn not terminating
        # and training jobs for RNNs not starting as a results,
        # I had to generate GP samples in a separate script.
        # Will find a way to fix this issue later on.
        bootstrapped_dataset = []

        for gpf in glob.glob(GP_BOOTSTRAP_SAMPLES_PATH)[::-1]:
            with open(gpf, "rb") as gpfp:
                bootstrapped_dataset.append(pickle.load(gpfp))

            if len(bootstrapped_dataset) == n_samples:
                break

    elif bootstrap_method == 'stationary_block':
        bootstrapped_dataset = stationary_boostrap_method(train_data.index,
                                                          train_data,
                                                          n_samples=n_samples)
    elif bootstrap_method == 'moving_block':
        bootstrapped_dataset = moving_block_bootstrap_method(train_data.index,
                                                             train_data,
                                                             n_samples=n_samples)
    elif bootstrap_method == 'circular_block':
        bootstrapped_dataset = circular_block_bootstrap_method(train_data.index,
                                                               train_data,
                                                               n_samples=n_samples)
    else:
        raise Exception("Bootstrap method not implemented!")

    transform_datasets = []

    for bt in bootstrapped_dataset:
        transform_datasets.append(timeseries_to_supervised(bt))

    return transform_datasets


def prepare_data(dataset, fraction_train=.7, bootstrap_method=None, n_samples=None):

    if dataset == "co2":
        dataset = read_mauna_loa_co2_data()
        train_dataset = dataset['CO2Concentration'][:int(fraction_train * len(dataset))]
        test_dataset = dataset['CO2Concentration'][int(fraction_train * len(dataset)):]
    elif dataset == "erie":
        dataset = read_lake_erie_data()
        train_dataset = dataset['Level'][:int(fraction_train * len(dataset))]
        test_dataset = dataset['Level'][int(fraction_train * len(dataset)):]
    else:
        raise Exception("No implementation for the requested dataset: " + str(dataset))

    if bootstrap_method is None:
        return train_dataset, test_dataset
    else:
        bootstrapped_train = bootstrap_data(train_dataset, bootstrap_method, n_samples)
        return train_dataset, test_dataset, bootstrapped_train


def plot_data(train_data, test_data, lstm_mean, lstm_sd):

    plt.plot(pd.concat([train_data, test_data]), color="blue")

    # Plot individual predictions:
    #  for pred in preds:
    #      plt.plot(pred, color="green", alpha=.1)

    plt.plot(lstm_mean, color="red")

    plt.fill_between(list(range(len(lstm_mean))),
                     lstm_mean - 2 * lstm_sd,
                     lstm_mean + 2 * lstm_sd,
                     color="gray", alpha=0.2)

    plt.axvline(len(train_data), color="black", linestyle='--')

    plt.title(DATASET_TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)

    plt.savefig("./figures/" + DATASET_TITLE + ".png")


def synthesize_predictions():

    preds = []

    for predf in glob.glob("./predictions/*.pkl"):
        with open(predf, "rb") as pf:
            preds.append(pickle.load(pf))

    if len(preds) < 1:
        raise Exception("Failed to generate predictions!")

    # line plot of observed vs predicted
    lstm_means = np.array([])
    lstm_sd = np.array([])

    for k in range(len(preds[0])):
        step_vals = [el[k] for el in preds]
        lstm_means = np.append(lstm_means, np.mean(step_vals))
        lstm_sd = np.append(lstm_sd, np.std(step_vals))

    return lstm_means, lstm_sd


def main():

    # ----------------------------------------
    # Step 1: Train the models in the ensemble
    if RETRAIN:

        train, test, bs_train = prepare_data(DATASET,
                                             bootstrap_method=BOOTSTRAP_METHOD,
                                             n_samples=ENSEMBLE_SIZE)

        # Delete old models & their performance scores:
        for mod_path in glob.glob("./models/" + DATASET + "/*.kmodel"):
            os.remove(mod_path)

        for mod_path in glob.glob("./model_performance/" + DATASET + "/*.pkl"):
            os.remove(mod_path)

        print "Training ensemble..."

        Parallel(n_jobs=N_JOBS)(delayed(train_model)(dat[0], dat[1], idx)
                                for idx, dat in enumerate(bs_train))

        print "Done training ensemble!"

    else:
        train, test = prepare_data(DATASET)

    # ----------------------------------------
    # Step 2: Generate predictions from trained models:

    # Delete old predictions
    for pred_path in glob.glob("./predictions/*.pkl"):
        os.remove(pred_path)

    # Load trained models' performance scores
    mod_pf = {}
    for modpf in glob.glob("./model_performance/" + DATASET + "/*.pkl"):
        with open(modpf, "rb") as modof:
            mod_pf[int(os.path.basename(modpf).replace(".pkl", ""))] = pickle.load(modof)

    # Select the best models:
    best_models = sorted(mod_pf.keys(), key=lambda x: mod_pf[x])[:int(np.ceil(BEST_PERFORMING_FRACTION * len(mod_pf)))]

    # Generate predictions from best models:
    Parallel(n_jobs=N_JOBS)(delayed(lstm_predict)(pd.concat([test, train]),
                                                  idx)
                            for idx in best_models)

    lstm_means, lstm_sd = synthesize_predictions()

    plot_data(train, test, lstm_means, lstm_sd)


if __name__ == "__main__":
    main()

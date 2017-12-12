from pyESN.pyESN import ESN
from utils.data import *
from utils.utils import *
from utils.plot import *
import pickle


def grid_search(x, y):
    x_train, y_train = x[:int(len(x) * 0.7)], y[:int(len(y) * 0.7)]
    x_test, y_test = x[int(len(x) * 0.7):], y[int(len(y) * 0.7):]
    n_res = list(range(5, 51, 5))
    s_rad = [0.1 * i for i in range(1, 101)]
    rng = list(range(1,501))
    min_mse = float("inf")
    best_par = [0] * 4
    for res in n_res:
        for rad in s_rad:
            for rs in rng:
                model = ESN(n_inputs=len(x_train[0]), n_outputs=1,
                            n_reservoir=res,
                            spectral_radius=rad,
                            random_state=rs)

                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                mse = np.sqrt(np.mean((pred.flatten() - y_test) ** 2))
                if mse < min_mse:
                    min_mse = mse
                    best_par = [res, rad, rs]

    return best_par


def weight_uncertainty(model, n_esns, x_train, y_train, x_test_len, seq_len, window_length, scaler):
    trainlen = len(y_train)
    future = x_test_len
    pred = np.zeros(shape=(n_esns, trainlen + future))
    for i in range(n_esns):
        pred_training = model.fit(x_train, y_train)
        prediction = predict_sliding(model, x_train, x_test_len, seq_len, window_length)
        prediction = invert_scale(scaler, prediction)
        prediction = prediction + trend[window_length + 1:]
        pred[i] = prediction
    # get mean & std over predictions
    stats = list(map(lambda i: (np.mean(pred[:, i]), np.std(pred[:, i])), range(trainlen + future)))
    mean = np.array([[tup[0]] for tup in stats])
    std = np.array([[tup[1]] for tup in stats])
    lb = [mean[i][0] - 2 * std[i][0] for i in range(trainlen + future)]
    ub = [mean[i][0] + 2 * std[i][0] for i in range(trainlen + future)]
    return mean, lb, ub


def model_uncertainty(res, x_train, y_train, x_test_len, seq_len, window_length, scaler, best):
    trainlen = len(y_train)
    future = x_test_len
    pred = np.zeros(shape=(len(res), trainlen + future))
    for i in range(len(res)):
        model = ESN(n_inputs=len(x_train[0]), n_outputs=1,
                    n_reservoir=res[i],
                    spectral_radius=best[1],
                    random_state=best[2])
        pred_training = model.fit(x_train, y_train)
        prediction = predict_sliding(model, x_train, x_test_len, seq_len, window_length)
        prediction = invert_scale(scaler, prediction)
        prediction = prediction + trend[window_length + 1:]
        pred[i] = prediction
    # get mean & std over predictions
    stats = list(map(lambda i: (np.mean(pred[:, i]), np.std(pred[:, i])), range(trainlen + future)))
    mean = np.array([[tup[0]] for tup in stats])
    std = np.array([[tup[1]] for tup in stats])
    lb = [mean[i][0] - 2 * std[i][0] for i in range(trainlen + future)]
    ub = [mean[i][0] + 2 * std[i][0] for i in range(trainlen + future)]
    return mean, lb, ub


def save_par(var, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var, f)


def load_par(filename):
    with open(filename, 'rb') as f:
        var = pickle.load(f)
    return var


# set parameters
window_length = 20
train_percent = 0.7
seq_len = 20

# load data
data = load_co2("data/mauna-loa-atmospheric-co2.csv")
# detrend, create windows over time points and normalize data
x_train, y_train, x_test, y_test, scaler, trend = create_direct_data(data, window_length, train_percent)
x_train = x_train.reshape(len(x_train), len(x_train[0]))
y_train = y_train.reshape(len(y_train))
x_test = x_test.reshape(len(x_test), len(x_test[0]))
y_test = y_test.reshape(len(y_test))
trainlen = len(y_train)
future = len(y_test)
# search for best number of hidden neurons, spectral radii & random state
best_par = grid_search(x_train, y_train)

# model uncertainty with variation in weights
model = ESN(n_inputs=len(x_train[0]), n_outputs=1,
            n_reservoir=best_par[0],
            spectral_radius=best_par[1],
            random_state=best_par[2])
mean, lb, ub = weight_uncertainty(model, 100, x_train, y_train, len(x_test), seq_len, window_length, scaler)

# model uncertainty with variation in number of hidden units
n_res = list(range(15, 36))
mean, lb, ub = model_uncertainty(n_res, x_train, y_train, len(x_test), seq_len, window_length, scaler, best_par)

y_true = np.concatenate((y_train, y_test))
y_true = invert_scale(scaler, y_true)
y_true = y_true + trend[window_length + 1:]
# plot results
plot_co2(y_true, mean, lb, ub, trainlen, best_par[0], best_par[1])

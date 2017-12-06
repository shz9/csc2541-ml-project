from model import *
from data import *
from plot import *
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

np.random.seed(111)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, dest="epochs")
parser.add_argument("--seq-len", type=int, default=50, dest="seq_len")
parser.add_argument("--seq-dim", type=int, default=3, dest="seq_dim")
parser.add_argument("--batch-size", type=int, default=100, dest="batch_size")
parser.add_argument("--samples", type=int, default=20, dest="samples")
parser.add_argument("--hidden-size", type=int, default=20, dest="hidden_size")


def train(x_train, y_train, x_test, y_test, args):
    model = RNN_many_to_one_stateless(args.batch_size, args.seq_len, args.seq_dim).model

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
              validation_data=[x_test, y_test])

    return model


def train_stateful(x_train, y_train, args):
    model = RNN_many_to_one_stateful(1, args.seq_len, args.seq_dim).model

    for i in range(args.epochs):
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False)
        model.reset_states()

    return model


def evaluate_model(model, x, args):
    y_pred = model.predict(x)

    return y_pred


def evaluate_model_test(model, x, args):
    cur_window = x[0]
    pred = []
    pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0, 0])

    for i in range(len(x) - 1):
        cur_window = x[i + 1]

        if i < cur_window.shape[0] - 1:
            cur_window[cur_window.shape[0] - i - 2: cur_window.shape[0] - 1, 1] = np.array(pred[len(pred) - i - 1:len(pred)]).reshape(-1)
        else:
            cur_window[0:cur_window.shape[0] - 1, 1] = np.array(pred[len(pred) - cur_window.shape[0] + 1:len(pred)]).reshape(-1)

        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0, 0])

    pred = np.array(pred)

    return pred


def evaluate_model_test_temp(model, x, args):
    cur_window = x[0]
    pred = []
    pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0, 0])

    for i in range(len(x) - 1):
        cur_window = x[i + 1]

        if i < cur_window.shape[0] - 1:
            cur_window[cur_window.shape[0] - i - 2: cur_window.shape[0] - 1, 2] = np.array(pred[len(pred) - i - 1:len(pred)]).reshape(-1)
        else:
            cur_window[0:cur_window.shape[0] - 1, 2] = np.array(pred[len(pred) - cur_window.shape[0] + 1:len(pred)]).reshape(-1)

        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0, 0])

    pred = np.array(pred)

    return pred


def temp(args):
    data_x_train, data_y_train = gen_data(-10, 10, args.samples, args.seq_len)
    data_x_test, data_y_test = gen_data(10, 30, args.samples, args.seq_len)

    scaler = MinMaxScaler()
    scaler.fit(data_x_train.reshape(-1, 1))
    data_x_train = scaler.transform(data_x_train.reshape(-1, 1)).reshape(-1)
    data_x_test = scaler.transform(data_x_test.reshape(-1, 1)).reshape(-1)

    diff_y_train = difference(data_y_train)
    window_x_train = window_inputs(data_x_train)
    diff_y_test = difference(data_y_test)
    window_x_test = window_inputs(data_x_test)

    x_train, y_train = final_data(window_x_train, diff_y_train, args.seq_len)
    x_test, y_test = final_data(window_x_test, diff_y_test, args.seq_len)

    model = train(x_train, y_train, x_test, y_test, args)

    pred = evaluate_model_test_temp(model, x_test, args)
    plot_temp(np.arange(len(y_test)), y_test, pred)


def main(args):
    x_train, y_train, x_test, y_test = generate_data_many_to_one(args.samples, args.seq_len)

    model = train(x_train, y_train, args)
    y_pred = evaluate_model_test(model, x_train, args)

    plot_predictions_window(x_train, y_train, y_pred)

    y_pred = evaluate_model_test(model, x_test, args)
    plot_predictions_window(x_test, y_test, y_pred)

if __name__ == "__main__":
    temp(parser.parse_args())
from model import RNN_many_to_one
from data import generate_data, generate_data_many_to_one
from plot import plot_predictions, plot_predictions_window
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50, dest="epochs")
parser.add_argument("--seq-len", type=int, default=50, dest="seq_len")
parser.add_argument("--seq-dim", type=int, default=2, dest="seq_dim")
parser.add_argument("--batch-size", type=int, default=100, dest="batch_size")
parser.add_argument("--samples", type=int, default=20, dest="samples")
parser.add_argument("--hidden-size", type=int, default=20, dest="hidden_size")


def train(x_train, y_train, args):
    model = RNN_many_to_one(args.batch_size, args.seq_len - 1, 2).model

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, shuffle=True)

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

def main(args):
    x_train, y_train, x_test, y_test = generate_data_many_to_one(args.samples, args.seq_len)

    model = train(x_train, y_train, args)
    y_pred = evaluate_model_test(model, x_train, args)

    plot_predictions_window(x_train, y_train, y_pred)

if __name__ == "__main__":
    main(parser.parse_args())
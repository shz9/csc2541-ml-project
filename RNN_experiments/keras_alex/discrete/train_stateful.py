from models import create_model_stateless, create_model_stateful
from utils.utils import evaluate_model_static, evaluate_model_sliding_stateful
from utils.data import *
from utils.plot import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=20, dest="seq_len")
parser.add_argument("--seq-dim", default=1, dest="seq_dim")
parser.add_argument("--train-percent", default=0.7, dest="train_percent")
parser.add_argument("--co2", default="data/mauna-loa-atmospheric-co2.csv", dest="co2")
parser.add_argument("--window-length", default=20, dest="window_length")
parser.add_argument("--batch-size", default=1, dest="batch_size")
parser.add_argument("--epochs", default=25, dest="epochs")
parser.add_argument("--static-predictions", default="results/stateful_static_predictions.png", dest="static_predictions")
parser.add_argument("--sliding-predictions", default="results/stateful_sliding_predictions.png", dest="sliding_predictions")

def train_stateful(x, y, args):
    idxs = np.arange(x.shape[0])
    train_idx = idxs[:int(args.train_percent * x.shape[0])]
    test_idx = idxs[int(args.train_percent * x.shape[0]):]

    x_train, y_train = x[train_idx,], y[train_idx,]
    x_test, y_test = x[test_idx,], y[test_idx,]

    model = create_model_stateful(args.batch_size, args.seq_len, args.seq_dim)

    for epoch in range(args.epochs):
        model.fit(x_train, y_train, epochs=1, batch_size=args.batch_size, shuffle=False, validation_data=(x_test, y_test))
        model.reset_states()

    return model


def main(args):
    data = load_co2(args.co2)

    time_x, time_y = create_window_data(data[1].values, args.window_length + 1)
    co2_x, co2_y = create_window_data(data[0].values, args.window_length + 1)
    co2_x, co2_y = scale_data(co2_x, co2_y, args.train_percent)

    model = train_stateful(co2_x, co2_y, args)
    y_pred = evaluate_model_static(model, co2_x, co2_y, args)
    plot_predictions(co2_x, co2_y, y_pred, "Stateful Static Predictions", args.static_predictions)

    y_pred = evaluate_model_sliding_stateful(model, co2_x, co2_y, args)
    plot_predictions(co2_x, co2_y, y_pred, "Stateful Sliding Predictions", args.sliding_predictions)


if __name__ == "__main__":
    main(parser.parse_args())
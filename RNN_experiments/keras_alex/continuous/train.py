from model import RNN
from data import generate_data
from plot import plot_predictions
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, dest="epochs")
parser.add_argument("--seq-len", type=int, default=50, dest="seq_len")
parser.add_argument("--seq-dim", type=int, default=2, dest="seq_dim")
parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
parser.add_argument("--samples", type=int, default=100, dest="samples")
parser.add_argument("--hidden-size", type=int, default=20, dest="hidden_size")


def train(x_train, y_train, args):
    model = RNN(args.batch_size, 1, 2).model
    num_batches = x_train.shape[0] / args.batch_size

    for epoch in range(args.epochs):
        mean_loss = []

        for i in range(num_batches):
            for j in range(x_train.shape[1]):
                loss = model.train_on_batch(np.expand_dims(x_train[i * args.batch_size: (i * args.batch_size) + args.batch_size, j, :], axis=1),
                                            np.expand_dims(np.array([y_train[i * args.batch_size: (i * args.batch_size) + args.batch_size, j]]).T, axis=1))

                mean_loss.append(loss)

            model.reset_states()
        print("error: ", np.mean(mean_loss))

    return model


def evaluate_model(model, x, args):
    y_pred = np.zeros((args.samples, args.seq_len))
    last = np.zeros((args.batch_size))

    model.reset_states()
    num_batches = x.shape[0] / args.batch_size

    for i in range(num_batches):
        for j in range(x.shape[1]):
            y_pred[i * args.batch_size: (i * args.batch_size) + args.batch_size, j] = model.predict_on_batch(np.expand_dims(np.array([x[i * args.batch_size: (i * args.batch_size) + args.batch_size, j, 0], last]).T, axis=1))[:, 0, 0]
            last = y_pred[i * args.batch_size: (i * args.batch_size) + args.batch_size, j]
        model.reset_states()


    return y_pred

def main(args):
    x_train, y_train, x_test, y_test = generate_data(args.samples, args.seq_len)

    model = train(x_train, y_train, args)
    y_pred = evaluate_model(model, x_train, args)

    print(y_pred)
    print(y_train)

    plot_predictions(x_train, y_train, y_pred)




if __name__ == "__main__":
    main(parser.parse_args())
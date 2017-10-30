import argparse
import numpy as np
import tensorflow as tf
from data import generate_data
from model import Model
from plot import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, dest="epochs")
parser.add_argument("--seq-len", type=int, default=50, dest="seq_len")
parser.add_argument("--seq-dim", type=int, default=1, dest="seq_dim")
parser.add_argument("--batch-size", type=int, default=47, dest="batch_size")
parser.add_argument("--samples", type=int, default=50, dest="samples")
parser.add_argument("--hidden-size", type=int, default=20, dest="hidden_size")


def train(x_train, y_train, args):
    x = tf.placeholder(dtype=tf.float32, shape=[args.seq_len, args.batch_size, args.seq_dim], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[args.seq_len, args.batch_size, args.seq_dim], name="y")
    state = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.hidden_size], name="state")
    model = Model(x, y, state)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    num_batches = x_train.shape[0] / args.batch_size

    for epoch in range(args.epochs):
        _current_state = np.zeros((args.batch_size, args.hidden_size), dtype=np.float32)
        loss_list = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx

            batch_x = x_train[np.arange(start_idx, x_train.shape[0], num_batches), ]
            batch_y = y_train[np.arange(start_idx, x_train.shape[0], num_batches), ]

            batch_x = np.transpose(batch_x, [1, 0, 2])
            batch_y = np.transpose(batch_y, [1, 0, 2])

            _total_loss, _train_step, _current_state = sess.run([model.loss, model.optimize, model.state],
                                                                feed_dict={model.x: batch_x, model.y:batch_y, model.state: _current_state})
            loss_list.append(_total_loss)

        if epoch % 10 == 0:
            print(np.mean(np.array(loss_list)))

    saver = tf.train.Saver()
    saver.save(sess, 'model/saved.ckpt')
    sess.close()

    return model


def evaluate_model(model, x, y, args):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'model/saved.ckpt')

    num_batches = x.shape[0] / args.batch_size

    _current_state = np.zeros((args.batch_size, args.hidden_size))
    y_pred = []

    num_batches = x.shape[0] / args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx

        batch_x = x[np.arange(start_idx, x.shape[0], num_batches), ]
        batch_y = y[np.arange(start_idx, x.shape[0], num_batches), ]

        batch_x = np.transpose(batch_x, [1, 0, 2])
        batch_y = np.transpose(batch_y, [1, 0, 2])

        temp, _current_state = sess.run([model.prediction_test, model.state],
                                        feed_dict={model.x: batch_x, model.y: batch_y, model.state: _current_state})
        y_pred.append(temp)

    return y_pred


def main(args):
    x_train, y_train, x_test, y_test = generate_data(args.samples, args.seq_len, args.seq_dim)
    model = train(x_train, y_train, args)

    y_pred = evaluate_model(model, x_train, y_train, args)
    y_pred = format_predictions(y_pred, args.seq_len, args.seq_dim)

    plot_predictions(x_train, y_train, y_pred)


if __name__ == "__main__":
    main(parser.parse_args())
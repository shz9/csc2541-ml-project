import functools
import tensorflow as tf


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    def __init__(self, x, y, state):
        self.x = x
        self.y = y
        self.state = state
        self.current_state = None
        self.seq_dim = int(self.x.get_shape()[2])
        self.seq_len = int(self.x.get_shape()[0])
        self.batch_size = int(self.state.get_shape()[0])
        self.hidden_size = int(self.state.get_shape()[1])

        self.W_xh = tf.get_variable(name="W_xh", dtype=tf.float32, shape=[self.seq_dim, self.hidden_size],
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.W_hh = tf.get_variable(name="W_hh", dtype=tf.float32, shape=[self.hidden_size, self.hidden_size],
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.W_hy = tf.get_variable(name="W_hy", dtype=tf.float32, shape=[self.hidden_size, self.seq_dim],
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.W_yh = tf.get_variable(name="W_yh", dtype=tf.float32, shape=[self.seq_dim, self.hidden_size],
                               initializer=tf.random_normal_initializer(stddev=0.2))

        self.b_xh = tf.get_variable(name="b_xh", dtype=tf.float32, shape=self.seq_dim,
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.b_hh = tf.get_variable(name="b_hh", dtype=tf.float32, shape=self.hidden_size,
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.b_hy = tf.get_variable(name="b_hy", dtype=tf.float32, shape=self.seq_dim,
                               initializer=tf.random_normal_initializer(stddev=0.2))
        self.b_yh = tf.get_variable(name="b_yh", dtype=tf.float32, shape=self.seq_dim,
                               initializer=tf.random_normal_initializer(stddev=0.2))

        self.prediction_train
        self.prediction_test
        self.optimize

    @lazy_property
    def prediction_train(self):
        self.current_state = self.state
        state_series = []
        y_pred = []
        o_to_h = None

        for i in range(0, self.x.get_shape()[0]):
            current_input = self.x[i]
            input_to_hidden = current_input * self.W_xh + self.b_xh
            hidden_to_hidden = tf.matmul(self.current_state, self.W_hh) + self.b_hh

            if o_to_h is None:
                o_to_h = tf.constant(0.0, shape=[self.batch_size, self.hidden_size])
            else:
                o_to_h = self.y[i] * self.W_yh + self.b_yh

            hidden_to_output = tf.matmul(self.current_state, self.W_hy) + self.b_hy
            new_state = tf.sigmoid(input_to_hidden + hidden_to_hidden + o_to_h)

            y_pred.append(hidden_to_output)
            state_series.append(new_state)
            self.current_state = new_state

        return y_pred

    @lazy_property
    def prediction_test(self):
        current_state = self.state
        state_series = []
        y_pred = []
        o_to_h = tf.constant(0.0, shape=[self.batch_size, self.hidden_size])

        for i in range(0, self.x.get_shape()[0]):
            current_input = self.x[i]
            input_to_hidden = current_input * self.W_xh + self.b_xh
            hidden_to_hidden = tf.matmul(current_state, self.W_hh) + self.b_hh

            new_state = tf.sigmoid(input_to_hidden + hidden_to_hidden + o_to_h)

            hidden_to_output = tf.matmul(current_state, self.W_hy) + self.b_hy
            o_to_h = hidden_to_output * self.W_yh + self.b_yh

            y_pred.append(hidden_to_output)
            state_series.append(new_state)
            current_state = new_state

        return y_pred

    @lazy_property
    def loss(self):
        listed_y = [self.y[i] for i in range(0, self.y.get_shape()[0])]
        losses = [tf.losses.mean_squared_error(pred, true) for pred, true in zip(self.prediction_train, listed_y)]
        total_loss = tf.reduce_sum(losses)

        return total_loss

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

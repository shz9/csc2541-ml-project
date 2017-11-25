"""
Author: Shadi Zabad
Date: November 2017
-------------------
Code below modifies implementation given in:
http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/getting_started.ipynb
"""


import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from edward.models import Normal


def build_co2_dataset(window_size=20):
    """
    Read the CO2 dataset and break it into sliding windows of <window_size>.
    The windows will serve as inputs to the BNN and the output will be the
    value of the next element in the sequence.
    """
    ex_dataset = pd.read_csv('../../data/mauna-loa-atmospheric-co2.csv',
                             header=None)
    ex_dataset.columns = ['CO2Concentration', 'Time']

    scaler = preprocessing.StandardScaler()
    np_scaled = scaler.fit_transform(ex_dataset)
    ex_dataset = pd.DataFrame(np_scaled, index=ex_dataset.index, columns=ex_dataset.columns)

    x = []
    y = []

    for idx in range(0, len(ex_dataset) - window_size, 1):
        x.append(ex_dataset['CO2Concentration'][idx:idx + window_size])
        y.append(ex_dataset['CO2Concentration'][idx + window_size])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return x, y


def neural_network(x, weights_biases):

    h = tf.tanh(tf.matmul(x, weights_biases[0][0]) + weights_biases[0][1])

    for lyr in weights_biases[1:-1]:
        h = tf.tanh(tf.matmul(h, lyr[0]) + lyr[1])

    h = tf.matmul(h, weights_biases[-1][0]) + weights_biases[-1][1]

    return tf.reshape(h, [-1])


def build_networks(n_inputs=5, n_samples=10, n_layers=None):

    x_, y_ = build_co2_dataset(n_inputs)

    if n_layers is None:
        alpha = 10.0
        n_layers = int(float(len(x_)) / (alpha * (n_inputs + 1))) + 1

    print "Creating neural networks with", str(n_layers + 1), "layers"

    weights_biases = []
    q_weights_biases = []
    w_b_dict = dict()

    for nl in range(n_layers):
        w_i = Normal(loc=tf.zeros([n_inputs, n_inputs]),
                     scale=tf.ones([n_inputs, n_inputs]))
        b_i = Normal(loc=tf.zeros(n_inputs),
                     scale=tf.ones(n_inputs))
        qw_i = Normal(loc=tf.Variable(tf.random_normal([n_inputs, n_inputs])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_inputs, n_inputs]))))
        qb_i = Normal(loc=tf.Variable(tf.random_normal([n_inputs])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_inputs]))))

        weights_biases.append((w_i, b_i))
        q_weights_biases.append((qw_i, qb_i))

        w_b_dict[w_i] = qw_i
        w_b_dict[b_i] = qb_i

    w_i = Normal(loc=tf.zeros([n_inputs, 1]),
                 scale=tf.ones([n_inputs, 1]))
    b_i = Normal(loc=tf.zeros(1),
                 scale=tf.ones(1))
    qw_i = Normal(loc=tf.Variable(tf.random_normal([n_inputs, 1])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_inputs, 1]))))
    qb_i = Normal(loc=tf.Variable(tf.random_normal([1])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

    weights_biases.append((w_i, b_i))
    q_weights_biases.append((qw_i, qb_i))
    w_b_dict[w_i] = qw_i
    w_b_dict[b_i] = qb_i

    y = Normal(loc=neural_network(x_, weights_biases),
               scale=0.1 * tf.ones(len(x_)))

    mus = tf.stack(
        [neural_network(x_, [(w.sample(), b.sample()) for w, b in q_weights_biases])
         for _ in range(n_samples)])

    return mus, {y: y_}, w_b_dict, y_


sess = ed.get_session()
tf.global_variables_initializer().run()

n_iterations = 5000  # Number of iterations
n_features = 2  # Number of inputs to the neural network

# Number of inputs is set to 5, we sample 10 times
mus, y_dict, w_dict, y_vals = build_networks(n_inputs=n_features,
                                             n_samples=10,
                                             n_layers=4)


inference = ed.KLqp(w_dict, data=y_dict)
inference.run(n_iter=n_iterations)

outputs = mus.eval()

mean_val = np.mean(outputs, axis=0)
sd_interval = np.std(outputs, axis=0)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Modeling CO2 Dataset with BNNs (Window size: %d)" % n_features)
ax.plot(list(range(len(outputs[0].T))), y_vals, 'ks', alpha=0.5)
ax.plot(list(range(len(outputs[0].T))), mean_val, 'r', lw=2, alpha=0.5)
ax.fill_between(list(range(len(outputs[0].T))),
                mean_val - 2 * sd_interval,
                mean_val + 2 * sd_interval,
                color="red", alpha=0.2)
ax.plot()
plt.show()

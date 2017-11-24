"""
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

    x = np.array(x, dtype=np.float32)   # .reshape((-1, 1))
    y = np.array(y, dtype=np.float32)
    return x, y


def neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])


D = 20   # number of features

x_train, y_train = build_co2_dataset(D)

W_0 = Normal(loc=tf.zeros([D, D]), scale=tf.ones([D, D]))
W_1 = Normal(loc=tf.zeros([D, D]), scale=tf.ones([D, D]))
W_2 = Normal(loc=tf.zeros([D, 1]), scale=tf.ones([D, 1]))
b_0 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b_1 = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = x_train
y = Normal(loc=neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2),
           scale=0.1 * tf.ones(len(x)))

qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, D])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, D]))))
qW_1 = Normal(loc=tf.Variable(tf.random_normal([D, D])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, D]))))
qW_2 = Normal(loc=tf.Variable(tf.random_normal([D, 1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 1]))))
qb_0 = Normal(loc=tf.Variable(tf.random_normal([D])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb_1 = Normal(loc=tf.Variable(tf.random_normal([D])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb_2 = Normal(loc=tf.Variable(tf.random_normal([1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


mus = tf.stack(
    [neural_network(x_train, qW_0.sample(), qW_1.sample(), qW_2.sample(),
                    qb_0.sample(), qb_1.sample(), qb_2.sample())
     for _ in range(10)])

sess = ed.get_session()

tf.global_variables_initializer().run()

"""
outputs = mus.eval()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(list(range(len(outputs[0].T))), outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(list(range(len(outputs[0].T))), outputs[1:].T, 'r', lw=2, alpha=0.5)
#ax.legend()
plt.show()"""

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={y: y_train})
inference.run(n_iter=30000)

# SECOND VISUALIZATION (posterior)

outputs = mus.eval()

mean_val = np.mean(outputs, axis=0)
sd_interval = np.std(outputs, axis=0)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Modeling CO2 Dataset with BNNs (Window size: 20)")
ax.plot(list(range(len(outputs[0].T))), y_train, 'ks', alpha=0.5)
ax.plot(list(range(len(outputs[0].T))), mean_val, 'r', lw=2, alpha=0.5)
ax.fill_between(list(range(len(outputs[0].T))),
                mean_val - 2 * sd_interval,
                mean_val + 2 * sd_interval,
                color="red", alpha=0.2)
ax.plot()
plt.show()

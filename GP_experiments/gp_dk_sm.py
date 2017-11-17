from __future__ import print_function
from data_loader import read_tp63_dataset
import numpy as np
import os
np.random.seed(42)
# Keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

# KGP
from kgp.models import Model
from kgp.layers import GP

# Dataset interfaces
from kgp.datasets.kin40k import load_data

# Model assembling and executing
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
from sklearn.datasets import fetch_mldata
import matplotlib.pylab as plt

def initCovSM(Q, D):
    w0 = np.log(np.ones((Q, 1)))
    mu = np.log(np.maximum(0.05 * np.random.rand(Q * D, 1), 1e-8))
    v = np.log(np.abs(np.random.randn(Q * D, 1) + 1))
    return [[w0], [mu], [v]]


def assemble_mlp(input_shape, batch_size, nb_train_samples):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(1024, activation='tanh', name='dense1')(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(512, activation='tanh', name='dense2')(hidden)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(64, activation='tanh', name='dense3')(hidden)
    hidden = Dropout(0.1)(hidden)
    hidden = Dense(2, activation='tanh', name='dense4')(hidden)

    gp = GP(hyp={
        'lik': np.log(0.3),
        'mean': np.zeros((2, 1)).tolist() + [[0]],
        'cov': initCovSM(4, 1),
    },
        inf='infGrid', dlik='dlikGrid',
        opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
        mean='meanSum', cov='covSM',
        update_grid=1,
        grid_kwargs={'eq': 1, 'k': 10.},
        cov_args=[4],
        mean_args=['{@meanLinear, @meanConst}'],
        batch_size=batch_size,
        nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]
    return Model(inputs=inputs, outputs=outputs)

tp63 = read_tp63_dataset()
x = np.array([[int(t)] for t in tp63.iloc[0].index.tolist()])
Y = tp63.iloc[0].values
y = []
for i in range(len(Y)):
    y.append([Y[i]])

y = np.array(y)
plt.plot(x, y)

x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]

data = {
    'train': [x_train, y_train],
    'valid': [x_test, y_test],
    'test': [x_test, y_test]
}

# Model & training parameters
input_shape = data['train'][0].shape[1:]
nb_train_samples = data['train'][0].shape[0]
output_shape = data['train'][1].shape[1:]
batch_size = 2 ** 10
epochs = 500

# Construct & compile the model
model = assemble_mlp(input_shape, batch_size, nb_train_samples=nb_train_samples)
loss = [gen_gp_loss(gp) for gp in model.output_layers]
model.compile(optimizer=Adam(1e-4), loss=loss)

# Train the model
history = train(model, data, callbacks=[], gp_n_iter=5,
                epochs=epochs, batch_size=batch_size, verbose=1)

# Test the model
x_test, y_test = data['test']
y_preds, y_var = model.predict(x_test, return_var=True)
rmse_predict = RMSE(y_test, y_preds)
print('Test RMSE:', rmse_predict)

plt.plot(x,y)
plt.plot(x_test, y_preds[0])

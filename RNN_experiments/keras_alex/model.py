from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.initializers import Constant, RandomNormal, RandomUniform
from keras.regularizers import l1


def create_model(seq_len, seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=None,
                 hidden_units=100, hidden_layers=1, activation="relu", regularization=0.01):
    x = Input(shape=(seq_len, seq_dim))

    if hidden_layers == 1:
        h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer)(x)
    else:
        h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer)(x)

    for i in range(1, hidden_layers):
        if i == hidden_layers - 1:
            h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer)(h)
        else:
            h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer)(h)

    y = Dense(seq_dim, activation = "linear", kernel_initializer=kernel_initializer)(h)

    model = Model(x, y)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model
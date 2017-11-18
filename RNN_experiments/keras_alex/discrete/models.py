from keras.layers import Dense, Input, LSTM, Dropout
from keras.models import Model
from keras.initializers import Constant, RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.regularizers import l1


def create_model_stateless(seq_len, seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=None,
                 hidden_units=50, hidden_layers=1, activation="tanh", regularization=0.01,
                 use_dropout=False, dropout_rate=0.001):
    x = Input(shape=(seq_len, seq_dim))

    if hidden_layers == 1:
        h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(x)
    else:
        h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(x)

    if use_dropout:
        h = Dropout(dropout_rate)(h)

    for i in range(1, hidden_layers):
        if i == hidden_layers - 1:
            h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(h)
        else:
            h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(h)

        if use_dropout:
            h = Dropout(dropout_rate)(h)

    y = Dense(seq_dim, activation = "linear", kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(h)

    model = Model(x, y)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def create_model_stateful(batch_size, seq_len, seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=None,
                 hidden_units=10, hidden_layers=1, activation="relu", regularization=0.01,
                 use_dropout=False, dropout_rate=0.0, learning_rate=0.001):
    x = Input(batch_shape=(batch_size, seq_len, seq_dim))

    if hidden_layers ==1:
        h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=False,
                 kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                 dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)
    else:
        h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=True,
                 kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                 dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)

    if use_dropout:
        h = Dropout(dropout_rate)(h)

    for i in range(1, hidden_layers):
        if i == hidden_layers - 1:
            h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=False,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                     dropout=dropout_rate, recurrent_dropout=dropout_rate)(h)
        else:
            h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=True,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                     dropout=dropout_rate, recurrent_dropout=dropout_rate)(h)

        if use_dropout:
            h = Dropout(dropout_rate)(h)

    y = Dense(seq_dim, activation="linear", kernel_initializer=kernel_initializer)(h)

    model = Model(x, y)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model
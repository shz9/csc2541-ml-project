from keras.layers import Dense, Input, LSTM, Dropout
from keras.models import Model, Sequential
from keras.initializers import Constant, RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.regularizers import l1, l2


def create_model_stateless(seq_len, seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=None,
                 hidden_units=50, hidden_layers=1, activation="tanh", regularization=0.01):
    x = Input(shape=(seq_len, seq_dim))

    if hidden_layers == 1:
        h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(x)
    else:
        h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(x)

    for i in range(1, hidden_layers):
        if i == hidden_layers - 1:
            h = LSTM(hidden_units, return_sequences = False, activation=activation, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(h)
        else:
            h = LSTM(hidden_units, return_sequences = True, activation=activation, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(h)

    y = Dense(seq_dim, activation = "linear", kernel_initializer=kernel_initializer, kernel_regularizer=None)(h)

    model = Model(x, y)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def create_model_stateful(batch_size, seq_len, seq_dim, kernel_initializer=RandomUniform(), kernel_regularizer=None,
                 hidden_units=10, hidden_layers=1, activation="relu"):
    x = Input(batch_shape=(batch_size, seq_len, seq_dim))

    if hidden_layers ==1:
        h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=False,
                 kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(x)
    else:
        h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=True,
                 kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(x)

    for i in range(1, hidden_layers):
        if i == hidden_layers - 1:
            h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=False,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(h)
        else:
            h = LSTM(hidden_units, activation=activation, stateful=True, return_sequences=True,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(h)


    y = Dense(seq_dim, activation="linear", kernel_initializer=kernel_initializer)(h)

    model = Model(x, y)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def create_dropout_rnn(seq_len, seq_dim, kernel_initializer=RandomNormal(), kernel_regularizer=l2(0.0001), hidden_units=100,
                       hidden_layers=1, activation="relu", dropout_recurrent=0.1, dropout_dense=0.1,
                       dropout_input=0.1, dropout_lstm=0.1):

    x = Input(shape=(seq_len, seq_dim))

    h = Dropout(dropout_input)(x)

    h = LSTM(hidden_units, activation=activation, kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
             recurrent_dropout=dropout_recurrent, dropout=dropout_lstm, bias_regularizer=kernel_regularizer,
             kernel_initializer=kernel_initializer, recurrent_initializer=kernel_initializer,
             bias_initializer=kernel_initializer)(h)

    h = Dropout(dropout_dense)(h)

    y = Dense(1, activation="linear", kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
              kernel_initializer=kernel_initializer, bias_initializer=kernel_initializer)(h)

    model = Model(x, y)

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def create_dropout_rnn_stateful(seq_len, seq_dim, kernel_initializer=RandomNormal(), kernel_regularizer=l2(0.0001), hidden_units=100,
                       hidden_layers=1, activation="relu", dropout_recurrent=0.1, dropout_dense=0.1,
                       dropout_input=0.1, dropout_lstm=0.1):
    x = Input(batch_shape=(1, seq_len, seq_dim))

    if hidden_layers == 1:
        h = Dropout(dropout_input)(x)

        h = LSTM(hidden_units, activation=activation, kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                 recurrent_dropout=dropout_recurrent, dropout=dropout_lstm, bias_regularizer=kernel_regularizer, stateful=True,
                 kernel_initializer=kernel_initializer, recurrent_initializer=kernel_initializer,
                 bias_initializer=kernel_initializer, return_sequences=False)(h)

    for i in range(hidden_layers - 1):
        h = LSTM(hidden_units, activation=activation, kernel_regularizer=kernel_regularizer,
                 recurrent_regularizer=kernel_regularizer,
                 recurrent_dropout=dropout_recurrent, dropout=dropout_lstm, bias_regularizer=kernel_regularizer,
                 stateful=True,
                 kernel_initializer=kernel_initializer, recurrent_initializer=kernel_initializer,
                 bias_initializer=kernel_initializer, return_sequences=True)(h)

    if hidden_layers > 1:
        h = LSTM(hidden_units, activation=activation, kernel_regularizer=kernel_regularizer,
                 recurrent_regularizer=kernel_regularizer,
                 recurrent_dropout=dropout_recurrent, dropout=dropout_lstm, bias_regularizer=kernel_regularizer,
                 stateful=True,
                 kernel_initializer=kernel_initializer, recurrent_initializer=kernel_initializer,
                 bias_initializer=kernel_initializer, return_sequences=False)(h)

    h = Dropout(dropout_dense)(h)

    y = Dense(1, activation="linear", kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
              kernel_initializer=kernel_initializer, bias_initializer=kernel_initializer)(h)

    model = Model(x, y)

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def reinstantiate_model(config, weights):
    model = Model.from_config(config)
    model.set_weights(weights)

    return model


# class CustomLSTM(LSTM):
#     def call(self, inputs, mask=None, training=None, initial_state=None):
#         self.cell._generate_dropout_mask(inputs, training=training)
#         self.cell._generate_recurrent_dropout_mask(inputs, training=training)
#         return RNN.call(inputs, mask=mask, training=training,
#                         initial_state=initial_state)
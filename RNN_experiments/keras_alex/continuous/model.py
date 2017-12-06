from highlevel.layers import Dense, Input, LSTM
from highlevel.models import Model
from highlevel.optimizers import Adam
from highlevel.regularizers import L1L2


class RNN():
    def __init__(self, batch_size, seq_len, seq_dim):
        x = Input(batch_shape=(batch_size, seq_len, seq_dim))

        h_1 = LSTM(100, activation="sigmoid", return_sequences=True, stateful=True)(x)

        y = Dense(1)(h_1)

        model = Model(x, y)
        model.compile(optimizer="adam", loss="mean_squared_error")

        self.model = model


class RNN_many_to_one_stateful():
    def __init__(self, batch_size, seq_len, seq_dim):
        x = Input(batch_shape=(1, seq_len, seq_dim))

        h_1 = LSTM(100, activation="relu", stateful=True, return_sequences=False)(x)
        # h_2 = LSTM(50, activation="sigmoid", return_sequences=False)(h_1)

        y = Dense(1, activation="linear")(h_1)

        model = Model(x, y)
        model.compile(optimizer=Adam(), loss="mean_squared_error")

        self.model = model


class RNN_many_to_one_stateless():
    def __init__(self, batch_size, seq_len, seq_dim):
        x = Input(shape=(seq_len, seq_dim))

        h_1 = LSTM(200, activation="tanh", return_sequences=False)(x)
        # h_2 = LSTM(50, activation="tanh", return_sequences=False)(h_1)

        y = Dense(1, activation="linear")(h_1)

        model = Model(x, y)
        model.compile(optimizer=Adam(), loss="mean_squared_error")

        self.model = model
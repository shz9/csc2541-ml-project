import keras.backend as K
from keras.models import Model
import numpy as np
import tensorflow as tf
from models import reinstantiate_model
np.set_printoptions(threshold=np.nan)


def mc_forward(config, weights, x_train, test_len, n_samples, state="stateless", full=True):
    model = reinstantiate_model(config, weights)

    model.reset_states()

    mc_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    learning_phase = True

    if state == "stateful":
        mc_samples_train = model.predict(x_train, batch_size=1)
        mc_samples_train = [mc_samples_train for _ in range(n_samples)]
        mc_samples_train = np.array(mc_samples_train)
    else:
        mc_samples_train = [mc_output([x_train, learning_phase])[0] for _ in range(n_samples)]
        mc_samples_train = np.array(mc_samples_train)

    pred_outer = []

    for j in range(len(mc_samples_train)):
        pred_inner = []
        cur_window = x_train[-1]

        for i in range(test_len):
            pred_inner.append(mc_output([np.expand_dims(cur_window, 0), learning_phase])[0])
            cur_window = cur_window[1:]
            cur_window = np.concatenate((cur_window, np.array(pred_inner[i])))

        pred_inner = np.array(pred_inner)
        pred_outer.append(pred_inner)

    pred_outer = np.array(pred_outer)

    if full:
        result = np.concatenate((mc_samples_train, pred_outer[:, :, :, 0]), axis=1)
    else:
        result = pred_outer

    return result


def mc_mean(samples):
    return np.mean(samples, 0)


def mc_std(samples):
    # term_1 = np.mean(samples ** 2, 0)
    # term_2 = np.mean(samples, 0) ** 2
    #
    # method1 = term_1 - term_2
    method2 = np.std(samples, axis=0)

    # print(method1)
    # print(method2)
    #
    # print(method1 - method2)

    return method2

    # return np.std(samples, axis=0)
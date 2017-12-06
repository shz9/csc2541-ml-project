import numpy as np

from utils.mc import mc_forward
from models import reinstantiate_model

def print_grid_results(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def predict_sliding(model, x_train, test_len, batch_size, seq_len, full=True):
    if isinstance(model, list):
        model = reinstantiate_model(model[0], model[1])

    model.reset_states()

    train_pred = model.predict(x_train, batch_size=batch_size)
    cur_window = train_pred[-seq_len:]
    pred = []

    for i in range(test_len):
        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0,0])
        cur_window = cur_window[1:]
        cur_window = np.concatenate((cur_window, np.array([[pred[i]]])))

    pred = np.array(pred)

    if full:
        result = np.concatenate((train_pred.reshape(-1), pred))
    else:
        result = pred

    return result


def evaluate_model_static(model, x, y, args):
    score = model.evaluate(x, y, batch_size=args.batch_size)
    model.reset_states()

    print('\n Score: ')
    print(score)

    pred = model.predict(x, batch_size=args.batch_size)

    return pred


def predict_different_start_points(config, weights, x_train, test_len, n_samples=5, state="stateless", full=True):
    state_points = 12
    results = []

    for i in range(1, state_points + 1):
        temp = mc_forward(config, weights, x_train[:-i], test_len + i, n_samples=n_samples, state=state, full=full)
        results.append(temp)

    results = np.array(results)

    return results


def pred_temp(model, x_train, test_len, args, full=True):
    state_points = 12
    results = []

    for i in range(1, state_points + 1):
        temp = predict_sliding(model, x_train[:-i], test_len + i, args)
        results.append(temp)

    results = np.array(results)

    return results
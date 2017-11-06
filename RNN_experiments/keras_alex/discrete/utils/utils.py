import numpy as np

def print_grid_results(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def evaluate_model_sliding_stateful(model, x, y, args):
    cur_window = x[0]
    pred = []

    for i in range(int(x.shape[0] * args.train_percent)):
        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0,0])
        cur_window = x[i]

    for i in range(int(x.shape[0] * args.train_percent), x.shape[0]):
        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0,0])
        cur_window = cur_window[1:]
        cur_window = np.concatenate((cur_window, np.array([[pred[i]]])))

    pred = np.array(pred)

    return pred


def evaluate_model_sliding_stateless(model, x, y):
    cur_window = x[0]
    pred = []

    for i in range(x.shape[0]):
        pred.append(model.predict(np.expand_dims(cur_window, axis=0))[0,0])
        cur_window = cur_window[1:]
        cur_window = np.concatenate((cur_window, np.array([[pred[i]]])))

    pred = np.array(pred)

    return pred


def evaluate_model_static(model, x, y, args):
    score = model.evaluate(x, y, batch_size=args.batch_size)
    model.reset_states()

    print('\n Score: ')
    print(score)

    pred = model.predict(x, batch_size=args.batch_size)

    return pred

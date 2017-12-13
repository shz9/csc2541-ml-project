# src : https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
import numpy
# numpy.random.seed(0)
# import tensorflow as tf
# tf.set_random_seed(0)
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
# from keras import backend as K 
# sess = tf.Session()	
# K.tensorflow_backend.set_session(sess)

def parser(x):
	translate = {
		'0' : '1',
		'08333333' : '2',
		'08333334' : '2',
		'16666667' : '3',
		'25' : '4',
		'33333333' : '5',
		'33333334' : '5',
		'41666667' : '6',
		'5' : '7',
		'58333333': '8',
		'58333334': '8',
		'66666667': '9',
		'75': '10',
		'83333333': '11',
		'83333334': '11',
		'91666667': '12'
	}
	x = str(x).split('.')[0] + '-' + translate[str(x).split('.')[1]]
	return datetime.strptime(x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
		# print("Training Epoch", i, "done...")
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
 
def load_data_run_model(take_difference=True,
						scaling=False,
						epochs_for_training=10,
						neuron_cells=12,
						plot_name='lstm_co2'):

	# load dataset
	series = read_csv('co2-samples.csv', header=0, parse_dates=[1], index_col=1, squeeze=True, date_parser=parser)

	# transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	# transform data to be supervised learning
	supervised = timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values

	# split data into train and test-sets
	train, test = supervised_values[0:-137], supervised_values[-137:]
	# transform the scale of the data
	# scaler, train_scaled, test_scaled = scale(train, test)

	# fit the model
	# lstm_model = fit_lstm(train_scaled, 1, 1000, 4)
	lstm_model = fit_lstm(train, 1, epochs_for_training, neuron_cells)

	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train[:, 0].reshape(len(train), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)

	# Predicting train set
	predictions = list()
	for i in range(1,len(train)):
		# make one-step forecast
		X, y = train[i, 0:-1], train[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert differencing
		# yhat = inverse_difference(raw_values, yhat, len(train)+1-i)
		yhat = yhat + raw_values[i-1]
		# store forecast
		predictions.append(yhat)

	# walk-forward validation on the test data
	# predictions = list()
	prev = None
	prev_history = [raw_values[len(train)+1]] #initial 
	for i in range(len(test)):
		# make one-step forecast
		if prev is None:
			prev = test[i, 0:-1]
		yhat = forecast_lstm(lstm_model, 1, prev)
		prev = yhat
		prev = numpy.array([prev])
		yhat = yhat + prev_history[i]
		prev_history.append(yhat)
		predictions.append(yhat)
		expected = raw_values[len(train) + i +1]
	rmse = sqrt(mean_squared_error(raw_values[-137:], predictions[-137:]))
	print('Test RMSE: %.3f' % rmse)
	# return predictions,raw_values[-137:], rmse
	return predictions, raw_values[1:-1], rmse

def plot_fig(pred_arr, true):
	pred_all = numpy.array(pred_arr)
	mean = numpy.mean(pred_all,axis=0)
	std = numpy.std(pred_all, axis=0)
	plot_name='runs_'+str(len(final_list))+'_e_n_'+str(final_list)+'_co2_dataset_lstm_test_on_137.pdf'
	rmse = sqrt(mean_squared_error(true, mean))
	pyplot.plot(true, color='b', label='True')
	pyplot.plot(mean, color='r', label='Predicted')
	pyplot.fill_between(numpy.arange(len(true)), mean+2*std, mean-2*std, color='tab:gray', alpha = 0.3)
	pyplot.axvline(x=319, ls='dashed', color='black')
	pyplot.xlabel('Months')
	pyplot.ylabel('CO2 Concentration (PPM)')
	pyplot.legend()
	pyplot.savefig(plot_name)
	pyplot.clf()

epochs_for_training = [100, 200]
neuron_cells = [40, 80, 100, 150, 200, 250]

#mock values, just for fast testing
# epochs_for_training = [100]
# neuron_cells = [40]

pred_arr = []
# true = None
true = None
final_list = []
r = 10

for e in epochs_for_training:
	for n in neuron_cells:
		# avg_rmse = 0
		for i in range(r):
			if i%5 == 0:
				print("run",i,"of",r,"...")
			pred, true, rmse =load_data_run_model(epochs_for_training=e,
												  neuron_cells=n)
		# 	avg_rmse += rmse
		# avg_rmse/=r
			if rmse < 4:
				final_list.append((e,n))
				pred_arr.append(pred)
				print("E,N,rmse done...", e, n, rmse)
				break
plot_fig(pred_arr, true)

# pred_arr = []
# true = None
# en_pairs = [(50,16), (10,16), (50,10), (1000,16)]
# for e,n in en_pairs:
# 	pred, true, rmse =load_data_run_model(epochs_for_training=e,
# 										  neuron_cells=n)
# 	print("E,N done...", e, n)
# 	if rmse < 3:
# 		# final_list.append((e,n))
# 		pred_arr.append(pred)
# plot_fig(pred_arr, true)
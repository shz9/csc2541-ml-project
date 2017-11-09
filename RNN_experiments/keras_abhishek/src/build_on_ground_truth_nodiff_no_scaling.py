# src : https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
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
import numpy
 
# date-time parsing function for loading the dataset
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
		print("Training Epoch", i, "done...")
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
 
# load dataset
series = read_csv('co2-samples.csv', header=0, parse_dates=[1], index_col=1, squeeze=True, date_parser=parser)

# transform data to be stationary
raw_values = series.values
# diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(raw_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-228], supervised_values[-228:]

# transform the scale of the data
# scaler, train_scaled, test_scaled = scale(train, test)
# just for easy of not having to rewrite the code
scaler, train_scaled, test_scaled = None, train, test

# fit the model
# lstm_model = fit_lstm(train_scaled, 1, 1000, 4)
lstm_model = fit_lstm(train_scaled, 1, 1000, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
prev = None
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	print( "X is : ", X)
	# if (!prev):
	# 	prev = test_scaled[0, 0:-1]
	yhat = forecast_lstm(lstm_model, 1, X)
	print('yhat after forecast: ', yhat)
	# invert scaling
	# yhat = invert_scale(scaler, X, yhat)
	# print('yhat after inverse scale: ', yhat)
	# invert differencing
	# yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# print('yhat after inverse diff: ', yhat)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
# report performance
rmse = sqrt(mean_squared_error(raw_values[-228:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-228:])
pyplot.plot(predictions)
pyplot.show()
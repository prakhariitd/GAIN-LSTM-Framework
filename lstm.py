# multivariate output stacked lstm example
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

# split a multivariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# # define input sequence
# in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))

file_name = 'data/pmu.csv'
data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
dataset = data_x[1000:2500,0]
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# choose a number of time steps
n_steps = 10
# convert into input/output
X, y = split_sequence(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
# model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu')) 
# model.add(Bidirectional(LSTM(5, activation='relu')))
model.add(Dense(n_features))
# adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X, y, epochs=15, verbose=1, validation_split=0.33)
# model.save('model')

#TESTING
test_data = dataset[1400:1500]
testX, testY = split_sequence(test_data, n_steps)
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
testPredict = model.predict(testX, verbose=1)
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# print (testPredict.shape)
# print (testY.shape)

test_data1 = dataset[1000:1100]
testX1, testY1 = split_sequence(test_data1, n_steps)

modifiedY = testY.reshape((len(testY),1))
for i in range(len(modifiedY)):
	modifiedY[i] = testY1[i]
perc1Y = [0]*len(testY)
for i in range(len(testY)):
	perc1Y[i] = abs(float(100*(testPredict[i] - modifiedY[i])/testPredict[i]))


dataset = data_x[1000:2500,1]
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# choose a number of time steps
n_steps = 10
# convert into input/output
X, y = split_sequence(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
# model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu')) 
# model.add(Bidirectional(LSTM(5, activation='relu')))
model.add(Dense(n_features))
# adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X, y, epochs=15, verbose=1, validation_split=0.33)
# model.save('model')

#TESTING
test_data = dataset[1400:1500]
testX, testY = split_sequence(test_data, n_steps)
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
testPredict = model.predict(testX, verbose=1)
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

test_data1 = dataset[1000:1100]
testX1, testY1 = split_sequence(test_data1, n_steps)

modifiedY = testY.reshape((len(testY),1))
for i in range(len(modifiedY)):
	modifiedY[i] = testY1[i]
perc2Y = [0]*len(testY)
for i in range(len(testY)):
	perc2Y[i] = abs(float(100*(testPredict[i] - modifiedY[i])/testPredict[i]))

dataset = data_x[1000:2500,2]
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# choose a number of time steps
n_steps = 10
# convert into input/output
X, y = split_sequence(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
# model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu')) 
# model.add(Bidirectional(LSTM(5, activation='relu')))
model.add(Dense(n_features))
# adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X, y, epochs=15, verbose=1, validation_split=0.33)
# model.save('model')

#TESTING
test_data = dataset[1400:1500]
testX, testY = split_sequence(test_data, n_steps)
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
testPredict = model.predict(testX, verbose=1)
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

test_data1 = dataset[1000:1100]
testX1, testY1 = split_sequence(test_data1, n_steps)

modifiedY = testY.reshape((len(testY),1))
for i in range(len(modifiedY)):
	modifiedY[i] = testY1[i]
perc3Y = [0]*len(testY)
for i in range(len(testY)):
	perc3Y[i] = abs(float(100*(testPredict[i] - modifiedY[i])/testPredict[i]))

# pyplot.plot(tdata, linestyle=':')
# pyplot.plot(idata)
pyplot.plot(perc1Y)
pyplot.plot(perc2Y)
pyplot.plot(perc3Y)
# pyplot.plot(,linestyle=':')
pyplot.ylabel('Percentage Error')
# pyplot.xlabel('Data point')
pyplot.legend(['Frequency', 'Voltage Magnitude', 'Voltage Angle'], loc='upper right')
pyplot.show()


# for i in range(len(testY)):
# 	stri1 = "diffs ["
# 	stri2 = "actual ["
# 	stri3 = "predict ["
# 	# if (diffY[i]>0.01*testY[i]):
# 	stri1 = stri1 + str(diffY[i]) + ", "
# 	stri2 = stri2 + str(testY[i]) + ", "
# 	stri3 = stri3 + str(testPredict[i]) + ", "
# 	stri1 = stri1 + "]"
# 	stri2 = stri2 + "]"
# 	stri3 = stri3 + "]"
# 	print (stri1)
# 	print (stri2)
# 	print (stri3)
# 	print()
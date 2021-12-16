from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

model = keras.models.load_model('model')

n_steps = 10
file_name = 'data/pmu.csv'
data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
dataset = data_x[:5000,:26]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

test_data = dataset[500:1000,:26]
testX, testY = split_sequences(test_data, n_steps)
testPredict = model.predict(testX, verbose=1)
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

diffY = np.zeros(testY.shape)
for i in range(testY.shape[0]):
	for j in range(testY.shape[1]):
		diffY[i][j] = abs(testY[i][j] - testPredict[i][j])

for i in range(testY.shape[0]):
	stri1 = "diffs ["
	stri2 = "actual ["
	stri3 = "predict ["
	for j in range(testY.shape[1]):
		if (diffY[i][j]>0.01*testY[i][j] and j!=1):
			stri1 = stri1 + str(diffY[i][j]) + ", "
			stri2 = stri2 + str(testY[i][j]) + ", "
			stri3 = stri3 + str(testPredict[i][j]) + ", "
	stri1 = stri1 + "]"
	stri2 = stri2 + "]"
	stri3 = stri3 + "]"
	print (stri1)
	print (stri2)
	print (stri3)
	print()
'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler
from utils import pmu_binary_sampler, pmu_forced_sampler
from keras.datasets import mnist


def data_loader (data_name, miss_rate):
	'''Loads datasets and introduce missingness.

	Args:
	- data_name: letter, spam, or mnist
	- miss_rate: the probability of missing components

	Returns:
	data_x: original data
	miss_data_x: data with missing values
	data_m: indicator matrix for missing components
	'''

	# Load data
	if data_name in ['letter', 'spam','pmu','pmu1','pmu2','pmu3','pmu4']:
		file_name = 'data/'+data_name+'.csv'
		data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
		# data_x = np.genfromtxt(file_name, delimiter=',', skip_header=True)[:, :] 
		# data_x = data_x[:10000,:]
	elif data_name == 'mnist':
		(data_x, _), _ = mnist.load_data()
		data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

	# Parameters
	no, dim = data_x.shape

	# Introduce missing data
	if (data_name=='pmu'):
		data_m = pmu_binary_sampler(1-miss_rate, no, dim)
		# data_m = pmu_forced_sampler(1-miss_rate, no, dim)
		miss_data_x = data_x.copy()
		miss_data_x[data_m == 0] = np.nan
	elif (data_name=='pmu2'):
		# data_m = pmu_binary_sampler(1-miss_rate, no, dim)
		data_m = pmu_forced_sampler(1-miss_rate, no, dim)
		miss_data_x = data_x.copy()
		miss_data_x[data_m == 0] = np.nan
	elif (data_name=='pmu3'):
		# data_m = pmu_binary_sampler(1-miss_rate, no, dim)
		data_m = pmu_forced_sampler(1-miss_rate, no, dim)
		miss_data_x = data_x.copy()
		miss_data_x[data_m == 0] = np.nan
	if (data_name=='pmu4'):
		data_m = pmu_binary_sampler(1-miss_rate, no, dim)
		# data_m = pmu_forced_sampler(1-miss_rate, no, dim)
		miss_data_x = data_x.copy()
	else:	
		data_m = binary_sampler(1-miss_rate, no, dim)
		miss_data_x = data_x.copy()
		miss_data_x[data_m == 0] = np.nan
	  
	return data_x, miss_data_x, data_m
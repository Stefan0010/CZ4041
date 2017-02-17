import pandas as pd
import numpy as np
import math

import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

def run(trainDataX, trainDataY, testDataX, testDataY, params):
	model = linear_model.Lasso(alpha = params['alpha'])

	print('Training starts...')
	startTime = time.time()
	model.fit(trainDataX, trainDataY)
	endTime = time.time()
	print('Training ended.')
	print('-----------------------------------')
	print('Time Elapsed : {} seconds'.format(endTime - startTime))

	yPredict = model.predict(testDataX)
	return math.sqrt(mse(testDataY, yPredict))

def runKFold(trainDataX, trainDataY, testDataX, testDataY, params, k):
	model = linear_model.LassoCV(cv = k)

	print('Training starts...')
	startTime = time.time()
	model.fit(trainDataX, trainDataY)
	endTime = time.time()
	print('Training ended.')
	print('-----------------------------------')
	print('Time Elapsed : {} seconds'.format(endTime - startTime))

	yPredict = model.predict(testDataX)
	return math.sqrt(mse(testDataY, yPredict))


# LOAD DATA
# extract both files to "splitted" folder OR
# update the path according to your local path
trainDataPath = '../../data/splitted/1000_split.csv'
valDataPath = '../../data/splitted/115_split.csv'

trainDataX = pd.read_csv(
				trainDataPath, 
				sep = ',', 
				header = 0, 
				dtype = {'StateHoliday' : np.dtype('a1')},
				usecols = [i for i in range(0,25) if i != 2 and i != 3 and i != 7]
				).values
trainDataY = pd.read_csv(trainDataPath, sep = ',', header = 0, usecols=[3]).values.flatten()

testDataX = pd.read_csv(
				valDataPath, 
				sep = ',', 
				header = 0, 
				dtype = {'StateHoliday' : np.dtype('a1')},
				usecols = [i for i in range(0,25) if i != 2 and i != 3 and i != 7]
				).values
testDataY = pd.read_csv(valDataPath, sep = ',', header = 0, usecols=[3]).values.flatten()

if np.any(trainDataX) and np.any(trainDataY) and np.any(testDataX) and np.any(testDataY):
	print("Training and Testing data loaded")
	print("-----------------------------------")
else:
	print("Failed to load Training and Testing data")


# PARAMETERS FOR LASSO REGRESSION
params = {
	'alpha' : 0.1,
	# 'max_iter' : 1000,
	'fit_intercept' : False
}

# NORMAL
rmse = run(trainDataX, trainDataY, testDataX, testDataY, params)

# KFOLD
# k = 2
# rmse = runKFold(trainDataX, trainDataY, testDataX, testDataY, params, k)

print('RMSE : {}'.format(rmse))
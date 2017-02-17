import pandas as pd
import numpy as np
import math

import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

def rmspe(yPred, yActual):
	filtr = (yActual != 0.).ravel()

	yActual = yActual[filtr].ravel()
	yPred = yPred[filtr].ravel()

	spe = ( (yActual - yPred) / yActual)**2
	rmspe = math.sqrt(np.mean(spe))

	return rmspe

def run(trainDataX, trainDataY, testDataX, testDataY, params):
	model = linear_model.Ridge(alpha = params['alpha'])

	print('Training starts...')
	startTime = time.time()
	model.fit(trainDataX, trainDataY)
	endTime = time.time()
	print('Training ended.')
	print('-----------------------------------')
	print('Time Elapsed : {} seconds'.format(endTime - startTime))

	yPredict = model.predict(testDataX)
	return rmspe(yPredict, testDataY)

def runKFold(trainDataX, trainDataY, testDataX, testDataY, params, k):
	model = linear_model.RidgeCV(cv = k)

	print('Training starts...')
	startTime = time.time()
	model.fit(trainDataX, trainDataY)
	endTime = time.time()
	print('Training ended.')
	print('-----------------------------------')
	print('Time Elapsed : {} seconds'.format(endTime - startTime))

	yPredict = model.predict(testDataX)
	return rmspe(yPredict, testDataY)

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
				usecols = [i for i in range(0,25) if i != 0 and i != 2 and i != 3 and i != 7]
				).values
trainDataY = pd.read_csv(trainDataPath, sep = ',', header = 0, usecols=[3]).values.flatten()

testDataX = pd.read_csv(
				valDataPath, 
				sep = ',', 
				header = 0, 
				dtype = {'StateHoliday' : np.dtype('a1')},
				usecols = [i for i in range(0,25) if i != 0 and i != 2 and i != 3 and i != 7]
				).values
testDataY = pd.read_csv(valDataPath, sep = ',', header = 0, usecols=[3]).values.flatten()

if np.any(trainDataX) and np.any(trainDataY) and np.any(testDataX) and np.any(testDataY):
	print("Training and Testing data loaded")
	print("-----------------------------------")
else:
	print("Failed to load Training and Testing data")

# PARAMETERS FOR RIDGE REGRESSION
params = {
	'alpha' : 0.1,
	# 'max_iter' : 1000,
	'fit_intercept' : False
}

# NORMAL
rmspe = run(trainDataX, trainDataY, testDataX, testDataY, params)

# KFOLD
# k = 5
# rmspe = runKFold(trainDataX, trainDataY, testDataX, testDataY, params, k)
print('RMSPE : {}'.format(rmspe))
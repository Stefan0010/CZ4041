import os
import pandas as pd
import numpy as np
import math

import time
import pickle
import util

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

def rmspe(y_pred, y_actual):
	filtr = (y_actual != 0.).ravel()

	y_actual = y_actual[filtr].ravel()
	y_pred = y_pred[filtr].ravel()

	spe = ( (y_actual - y_pred) / y_actual)**2
	rmspe = math.sqrt(np.mean(spe))

	return rmspe

def loadData(data_dir='../data/'):
	data = util.load_train_data(data_dir).values

	# split 2 last months for testing
	data_train = np.array([
		d for d in data 
		if not (d[2].year == 2015 and (d[2].month == 6 or d[2].month == 7))
		])

	data_test = np.array([d for d in data if d[2].year == 2015 and (d[2].month == 6 or d[2].month == 7)])

	x_train = np.delete(data_train, [0,2,3,4,7], axis=1)
	y_train = data_train[:,3]

	x_test = np.delete(data_test, [0,2,3,4,7], axis=1)
	y_test = data_test[:,3]

	return x_train, y_train, x_test, y_test

def train(x, y, model_name):
	# load model
	with open( os.path.join('lasso/', model_name), 'rb') as infile:
		model = pickle.load(infile)

	if model:
		print('Training starts...')
		startTime = time.time()
		model.fit(x, y)
		endTime = time.time()
		print('Training ended.')
		print('-----------------------------------')
		print('Time Elapsed : {} seconds'.format(endTime - startTime))

		with open( os.path.join('lasso/', model_name), 'wb') as outfile:
			pickle.dump(model, outfile)

		print('model saved!')
	else:
		print('model does not exist!')

def test(x, y, model_name):
	with open( os.path.join('lasso/', model_name), 'rb') as infile:
		model = pickle.load(infile)

	y_predict = model.predict(x)
	result = rmspe(y_predict, y)

	print('RMSPE : {}'.format(result))




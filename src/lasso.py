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

	x_train = np.delete(data_train, [0,2,3,4,7,25], axis=1)
	y_train = data_train[:,3]
	std_train = np.std(data_train[:,25])
	y_train_norm = (data_train[:,3] - data_train[:,25]) / std_train

	x_test = np.delete(data_test, [0,2,3,4,7,25], axis=1)
	y_test = data_test[:, 3]
	y_test_norm = data_test[:, [3, 25]]

	return x_train, y_train_norm, x_test, y_test_norm

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

def test_norm(x, y, model_name):
	with open( os.path.join('lasso/', model_name), 'rb') as infile:
		model = pickle.load(infile)

	y_predict = model.predict(x)
	std_test = np.std(y[:,1])
	y_predict_denorm = (y_predict * std_test) + y[:,1]

	result = rmspe(y_predict_denorm, y[:,0])
	print('RMSPE : {}'.format(result))

def evaluate():
	# TRAINING
	data_train = util.load_train_data('../data/').values
	x_train = np.delete(data_train, [0,2,3,4,7,25], axis=1)
	y_train = data_train[:,3]
	std_train = np.std(data_train[:,25])
	mean_train = data_train[:,25]
	y_train_norm = (data_train[:,3] - mean_train) / std_train

	model = linear_model.Lasso(alpha=1.0)
	model.fit(x_train, y_train_norm)

	# TESTING
	data_test = util.load_test_data('../data/').values
	x_test = np.delete(data_test, [0,1,3,6,24], axis=1)
	label = data_test[:,0]
	mean_test = data_test[:,24]
	std_test = np.std(data_test[:,24])
	y_predict = model.predict(x_test)
	y_predict_denorm = (y_predict * std_test) + mean_test

	result = pd.DataFrame(np.c_[label, y_predict_denorm])
	result.to_csv('../data/lasso_result.csv', header=['Id', 'Sales'], index=False)

	return True

# MAIN
# x_train, y_train_norm, x_test, y_test_norm = loadData()
# model = linear_model.Lasso(alpha=1.0)
# model.fit(x_train, y_train_norm)

# y_predict = model.predict(x_test)
# std_test = np.std(y_test_norm[:,1])
# y_predict_denorm = (y_predict * std_test) + y_test_norm[:,1]

# result = rmspe(y_predict_denorm, y_test_norm[:,0])
# print('RMSPE : {}'.format(result))

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
	std_train = data_train[:,26]
	mean_train = data_train[:,25]
	y_train_norm = (y_train - mean_train) / std_train

	x_test = np.delete(data_test, [0,2,3,4,7,25], axis=1)
	mean_test = data_test[:,25]
	std_test = data_test[:,26]
	y_test = data_test[:,3]

	return x_train, y_train_norm, mean_test, std_test, x_test, y_test

def evaluate():
	# TRAINING
	data_train = util.load_train_data('../data/').values
	x_train = np.delete(data_train, [0,2,3,4,7,25], axis=1)
	y_train = data_train[:,3]
	std_train = data_train[:,26]
	mean_train = data_train[:,25]
	y_train_norm = (y_train - mean_train) / std_train

	model = linear_model.Ridge(alpha=1.0)
	model.fit(x_train, y_train_norm)

	# TESTING
	data_test = util.load_test_data('../data/').values
	x_test = np.delete(data_test, [0,1,3,6,24], axis=1)
	label = data_test[:,0]
	mean_test = data_test[:,24]
	std_test = data_test[:,25]
	y_predict = model.predict(x_test)
	y_predict_denorm = (y_predict * std_test) + mean_test

	result = pd.DataFrame(np.c_[label, y_predict_denorm])
	result.to_csv('../data/ridge_result.csv', header=['Id', 'Sales'], index=False)

	return True

def predictPre():
	with open ('../data/dset.pickle', 'rb') as file:
		d = pickle.load(file)

	model = linear_model.Ridge(alpha=1.0)
	model.fit(d['xtrain'], d['ytrain'])

	y_pred = model.predict(d['xtest'])
	y_pred_denorm = (y_pred * d['std']) + d['mean']

	result = np.c_[d['submid'], y_pred_denorm]

	for i in d['closed']:
		result = np.vstack((result, [i, float(0)]))

	result = result[result[:,0].argsort()]
	result = pd.DataFrame(result)
	result[0] = result[0].astype(int)

	result.to_csv('../data/ridge_result_pre_2.csv', header=['Id', 'Sales'], index=False)

	return True

def predictPerStore():
	# initialize result array
	result = np.zeros(shape=(41088,2))

	tr = util.load_train_data('../data/')
	ts = util.load_test_data('../data/')

	ts_id = ts['Store'].unique()
	for i in ts_id:
		d_tr = tr[tr['Store'] == i]

		# x train
		x_tr = d_tr.copy()
		del x_tr['Store']
		del x_tr['Date']
		del x_tr['Sales']
		del x_tr['Customers']
		del x_tr['StateHoliday']
		del x_tr['Mean']
		del x_tr['Std']

		# normalized y train
		mean = d_tr['Mean']
		std = d_tr['Std']
		y_tr = d_tr['Sales']

		y_tr_norm = (y_tr - mean) / std

		# train model
		print('training for store id : {}'.format(i))
		model = linear_model.Ridge(alpha=1.0)
		model.fit(x_tr, y_tr_norm)

		# predict
		print('predicting for store id : {}'.format(i))

		d_ts = ts[ts['Store'] == i]

		# check for open or close
		# predict only for open store
		opened = d_ts[d_ts['Open'] == 1]
		closed = d_ts[d_ts['Open'] == 0]

		# x test
		x_ts = opened.copy()
		del x_ts['Id']
		del x_ts['Store']
		del x_ts['Date']
		del x_ts['StateHoliday']
		del x_ts['Mean']
		del x_ts['Std']

		# sales predict
		y_pred = model.predict(x_ts)

		# denom
		y_pred_denorm = (y_pred * opened['Std']) + opened['Mean']

		for j in opened['Id']:
			result[j-1] = [j, y_pred_denorm[j-1]]

		for k in closed['Id']:
			result[k-1] = [k, 0]

		print('result stored!')
		print('-------------------------------')
	
	result = pd.DataFrame(result)
	result[0] = result[0].astype(int)
	result.to_csv('../data/ridge_result_per_store.csv', header=['Id', 'Sales'], index=False)

	return True


# MAIN
# x_train, y_train_norm, mean_test, std_test, x_test, y_test = loadData()
# model = linear_model.Ridge(alpha=1.0)
# model.fit(x_train, y_train_norm)

# y_predict = model.predict(x_test)
# y_predict_denorm = (y_predict * std_test) + mean_test

# result = rmspe(y_predict_denorm, y_test)
# print('RMSPE : {}'.format(result))

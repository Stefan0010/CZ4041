import os
import pandas as pd
import numpy as np
import math

import time
import pickle
import util
from datetime import datetime

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
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
	with open ('../data/dset_ridge.pickle', 'rb') as file:
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

def preProcess(tr, ts):
	# day of week feature
	# greatly improve
	tr.insert(len(tr.columns), 'Monday', tr['DayOfWeek'] == 1)
	tr.insert(len(tr.columns), 'Tuesday', tr['DayOfWeek'] == 2)
	tr.insert(len(tr.columns), 'Wednesday', tr['DayOfWeek'] == 3)
	tr.insert(len(tr.columns), 'Thursday', tr['DayOfWeek'] == 4)
	tr.insert(len(tr.columns), 'Friday', tr['DayOfWeek'] == 5)
	tr.insert(len(tr.columns), 'Saturday', tr['DayOfWeek'] == 6)
	tr.insert(len(tr.columns), 'Sunday', tr['DayOfWeek'] == 7)

	ts.insert(len(ts.columns), 'Monday', ts['DayOfWeek'] == 1)
	ts.insert(len(ts.columns), 'Tuesday', ts['DayOfWeek'] == 2)
	ts.insert(len(ts.columns), 'Wednesday', ts['DayOfWeek'] == 3)
	ts.insert(len(ts.columns), 'Thursday', ts['DayOfWeek'] == 4)
	ts.insert(len(ts.columns), 'Friday', ts['DayOfWeek'] == 5)
	ts.insert(len(ts.columns), 'Saturday', ts['DayOfWeek'] == 6)
	ts.insert(len(ts.columns), 'Sunday', ts['DayOfWeek'] == 7)

	# date feature
	# ---- doesnt improve anth ----
	# tr.insert(len(tr.columns), 'Date_Norm', (tr['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
	# tr.insert(len(tr.columns), 'Day_Norm', (tr['Date'].dt.day - 1.) / 31.)
	# tr.insert(len(tr.columns), 'Month_Norm', (tr['Date'].dt.month - 1.) / 12.)
	# tr.insert(len(tr.columns), 'Year_Norm', (tr['Date'].dt.year - 2013.) / 2.)

	# ts.insert(len(ts.columns), 'Date_Norm', (ts['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
	# ts.insert(len(ts.columns), 'Day_Norm', (ts['Date'].dt.day - 1.) / 31.)
	# ts.insert(len(ts.columns), 'Month_Norm', (ts['Date'].dt.month - 1.) / 12.)
	# ts.insert(len(ts.columns), 'Year_Norm', (ts['Date'].dt.year - 2013.) / 2.)

	# ---- doesnt improve anth ----
	# tr.insert(len(tr.columns), 'Timestamp', pd.to_datetime(tr['Date']).values.astype(int))
	# ts.insert(len(ts.columns), 'Timestamp', pd.to_datetime(ts['Date']).values.astype(int))

	store_ids = tr['Store'].unique().astype(int, copy=False)

	# Sales
	tr.insert(len(tr.columns), 'Sales_Avg', 0.)
	tr.insert(len(tr.columns), 'Sales_Max', 0.)
	tr.insert(len(tr.columns), 'Sales_Min', 0.)

	ts.insert(len(ts.columns), 'Sales_Avg', 0.)
	ts.insert(len(ts.columns), 'Sales_Max', 0.)
	ts.insert(len(ts.columns), 'Sales_Min', 0.)
	
	for store_id in store_ids:
		mask = tr['Store'] == store_id
		mask2 = ts['Store'] == store_id
		ser = tr.loc[mask, 'Sales']

		val = ser.mean()
		tr.loc[mask, 'Sales_Avg'] = val
		ts.loc[mask2, 'Sales_Avg'] = val

		val = ser.max()
		tr.loc[mask, 'Sales_Max'] = val
		ts.loc[mask2, 'Sales_Max'] = val

		val = ser.min()
		tr.loc[mask, 'Sales_Min'] = val
		ts.loc[mask2, 'Sales_Min'] = val

	# Customers
	tr.insert(len(tr.columns), 'Customers_Avg', 0.)
	tr.insert(len(tr.columns), 'Customers_Max', 0.)
	tr.insert(len(tr.columns), 'Customers_Min', 0.)

	ts.insert(len(ts.columns), 'Customers_Avg', 0.)
	ts.insert(len(ts.columns), 'Customers_Max', 0.)
	ts.insert(len(ts.columns), 'Customers_Min', 0.)

	for store_id in store_ids:
		mask = tr['Store'] == store_id
		mask2 = ts['Store'] == store_id
		ser = tr.loc[mask, 'Customers']

		val = ser.mean()
		tr.loc[mask, 'Customers_Avg'] = val
		ts.loc[mask2, 'Customers_Avg'] = val

		val = ser.max()
		tr.loc[mask, 'Customers_Max'] = val
		ts.loc[mask2, 'Customers_Max'] = val

		val = ser.min()
		tr.loc[mask, 'Customers_Min'] = val
		ts.loc[mask2, 'Customers_Min'] = val

	return tr,ts

def trainKFold(dtrain, k):

	size = int(len(dtrain) / k)

	models = []
	scores = []

	for i in range(k):
		traind = dtrain[:i*size].append(dtrain[(i+1)*size:], ignore_index=True)
		vald = dtrain[i*size:][:size]

		# training data with validation
		# x train
		x_tr = traind.copy()
		del x_tr['Store']
		del x_tr['Date']
		del x_tr['DayOfWeek']
		del x_tr['Sales']
		del x_tr['Customers']
		del x_tr['StateHoliday']
		del x_tr['Mean']
		del x_tr['Std']

		# normalized y train
		mean_tr = traind['Mean']
		std_tr = traind['Std']
		y_tr = traind['Sales']
		y_tr_norm = (y_tr - mean_tr) / std_tr

		# val data
		# x val
		x_val = vald.copy()
		del x_val['Store']
		del x_val['Date']
		del x_val['DayOfWeek']
		del x_val['Sales']
		del x_val['Customers']
		del x_val['StateHoliday']
		del x_val['Mean']
		del x_val['Std']

		# normalized y val
		mean_val = vald['Mean']
		std_val = vald['Std']
		y_val = vald['Sales']
		y_val_norm = (y_val - mean_val) / std_val

		model = linear_model.Ridge(alpha=1.0)
		model.fit(x_tr, y_tr_norm)

		models.append(model)
		scores.append(model.score(x_val, y_val_norm))

	# return model with best score
	return models[scores.index(max(scores))]

def predictPerStore():
	# initialize result array
	result = np.zeros(shape=(41088,2))

	traind = util.load_train_data('../data/')
	testd = util.load_test_data('../data/')

	# additional features
	tr, ts = preProcess(traind, testd)

	ts_id = ts['Store'].unique()
	for i in ts_id:
		d_tr = tr[tr['Store'] == i]
		
		# train using kfold
		print('training for store id : {}'.format(i))
		k = 5
		model = trainKFold(d_tr, k)

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
		del x_ts['DayOfWeek']
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
	result.to_csv('../data/ridge_result_per_store_5_fold_6.csv', header=['Id', 'Sales'], index=False)

	return True

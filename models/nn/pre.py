import os
import sys

import caffe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pickle

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

# Some constants
batch_size = 10000
val_size = 50000

# Load train & test data
data_dir = '../../data'

traind = util.load_train_data(data_dir)
testd = util.load_test_data(data_dir)

def znorm(traind, testd, column):
    mean = traind[column].mean()
    std = traind[column].std()
    traind.loc[:, column] = (traind[column] - mean) / std
    testd.loc[:, column] = (testd[column] - mean) / std

# Sort
traind.sort_values(['Date', 'Store'], ascending=True, inplace=True)
testd.sort_values(['Date', 'Store'], ascending=True, inplace=True)

# Drop all when stores are closed
traind = traind[traind['Open'] == 1]
days_open = testd['Open'] == 1

# Store
store_ids = traind['Store'].unique().astype(int, copy=False)

# DayOfWeek
traind.insert(len(traind.columns), 'Monday', traind['DayOfWeek'] == 1)
traind.insert(len(traind.columns), 'Tuesday', traind['DayOfWeek'] == 2)
traind.insert(len(traind.columns), 'Wednesday', traind['DayOfWeek'] == 3)
traind.insert(len(traind.columns), 'Thursday', traind['DayOfWeek'] == 4)
traind.insert(len(traind.columns), 'Friday', traind['DayOfWeek'] == 5)
traind.insert(len(traind.columns), 'Saturday', traind['DayOfWeek'] == 6)
traind.insert(len(traind.columns), 'Sunday', traind['DayOfWeek'] == 7)

testd.insert(len(testd.columns), 'Monday', testd['DayOfWeek'] == 1)
testd.insert(len(testd.columns), 'Tuesday', testd['DayOfWeek'] == 2)
testd.insert(len(testd.columns), 'Wednesday', testd['DayOfWeek'] == 3)
testd.insert(len(testd.columns), 'Thursday', testd['DayOfWeek'] == 4)
testd.insert(len(testd.columns), 'Friday', testd['DayOfWeek'] == 5)
testd.insert(len(testd.columns), 'Saturday', testd['DayOfWeek'] == 6)
testd.insert(len(testd.columns), 'Sunday', testd['DayOfWeek'] == 7)

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  7, 31)
date_normer = float((date_max - date_min).days)

traind.insert(len(traind.columns), 'Date_Norm', (traind['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
testd.insert(len(testd.columns), 'Date_Norm', (testd['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)

# Sales
traind.insert(len(traind.columns), 'Sales_Avg', 0.)
traind.insert(len(traind.columns), 'Sales_Max', 0.)
traind.insert(len(traind.columns), 'Sales_Min', 0.)

testd.insert(len(testd.columns), 'Sales_Avg', 0.)
testd.insert(len(testd.columns), 'Sales_Max', 0.)
testd.insert(len(testd.columns), 'Sales_Min', 0.)

for store_id in store_ids:
    mask = traind['Store'] == store_id
    mask2 = testd['Store'] == store_id
    ser = traind.loc[mask, 'Sales']

    val = ser.mean()
    traind.loc[mask, 'Sales_Avg'] = val
    testd.loc[mask2, 'Sales_Avg'] = val

    val = ser.max()
    traind.loc[mask, 'Sales_Max'] = val
    testd.loc[mask2, 'Sales_Max'] = val

    val = ser.min()
    traind.loc[mask, 'Sales_Min'] = val
    testd.loc[mask2, 'Sales_Min'] = val

mean = traind['Sales'].mean()
std = traind['Sales'].std()
traind.loc[:, 'Sales'] = (traind.loc[:, 'Sales'] - mean) / std

znorm(traind, testd, 'Sales_Avg')
znorm(traind, testd, 'Sales_Max')
znorm(traind, testd, 'Sales_Min')

# Customers
traind.insert(len(traind.columns), 'Customers_Avg', 0.)
traind.insert(len(traind.columns), 'Customers_Max', 0.)
traind.insert(len(traind.columns), 'Customers_Min', 0.)

testd.insert(len(testd.columns), 'Customers_Avg', 0.)
testd.insert(len(testd.columns), 'Customers_Max', 0.)
testd.insert(len(testd.columns), 'Customers_Min', 0.)

for store_id in store_ids:
    mask = traind['Store'] == store_id
    mask2 = testd['Store'] == store_id
    ser = traind.loc[mask, 'Customers']

    val = ser.mean()
    traind.loc[mask, 'Customers_Avg'] = val
    testd.loc[mask2, 'Customers_Avg'] = val

    val = ser.max()
    traind.loc[mask, 'Customers_Max'] = val
    testd.loc[mask2, 'Customers_Max'] = val

    val = ser.min()
    traind.loc[mask, 'Customers_Min'] = val
    testd.loc[mask2, 'Customers_Min'] = val

del traind['Customers']

znorm(traind, testd, 'Customers_Avg')
znorm(traind, testd, 'Customers_Max')
znorm(traind, testd, 'Customers_Min')

# Open

# Promo

# StateHoliday
del traind['StateHoliday']
del testd['StateHoliday']

# SchoolHoliday
# StateHoliday_0
# StateHoliday_a
# StateHoliday_b
# StateHoliday_c

# Weekends
del traind['Weekends']
del testd['Weekends']

# Weekdays
del traind['Weekdays']
del testd['Weekdays']

# StoreType_a
# StoreType_b
# StoreType_c
# StoreType_d
# Assortment_a
# Assortment_b
# Assortment_c

# HasCompetition
traind.insert(len(traind.columns), 'NoCompetition', traind['HasCompetition'] == 0)
testd.insert(len(testd.columns), 'NoCompetition', testd['HasCompetition'] == 0)

# CompetitionDistance
max_compdist = traind['CompetitionDistance'].max()
traind.loc[:, 'CompetitionDistance'] = traind['CompetitionDistance'] / max_compdist
testd.loc[:, 'CompetitionDistance'] = testd['CompetitionDistance'] / max_compdist

traind.loc[traind['NoCompetition'], 'CompetitionDistance'] = 1.0
testd.loc[testd['NoCompetition'], 'CompetitionDistance'] = 1.0

# IsDoingPromo2

# Store ids changed from 1-1115 to 0-1114 for embedding layer
idvec = traind['Store'].as_matrix().astype(int, copy=False) - 1
x = traind.drop(['Store', 'DayOfWeek', 'Date', 'Sales', 'Open'], axis=1).as_matrix().astype(float, copy=False)
y = traind['Sales'].as_matrix().astype(float, copy=False)

shuffledidx = np.arange(len(idvec))
np.random.shuffle(shuffledidx)
idvec = idvec[shuffledidx]
x = x[shuffledidx]
y = y[shuffledidx]

idtrain = idvec[:-val_size]
xtrain = x[:-val_size]
ytrain = y[:-val_size]

idval = idvec[-val_size:]
xval = x[-val_size:]
yval = y[-val_size:]

prev_len = len(xtrain)
num_rem = batch_size - prev_len % batch_size
idtrain = np.insert(idtrain, prev_len, idtrain[:num_rem], 0)
xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)
ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)

# Store ids in test dataset changed from 1-1115 to 0-1114
idtest = testd.loc[days_open, 'Store'].as_matrix().astype(int, copy=False) - 1
submid = testd.loc[days_open, 'Id'].as_matrix().astype(int, copy=False)
xtest = testd.loc[days_open].drop(['Id', 'Store', 'DayOfWeek', 'Date', 'Open'], axis=1).as_matrix().astype(float, copy=False)
closed = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

prev_len = len(xtest)
num_rem = batch_size - prev_len % batch_size
idtest = np.insert(idtest, prev_len, idtest[:num_rem], 0)
xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)

with open('dset.pickle', 'wb') as f:
    pickle.dump({
        'mean': mean,
        'std': std,
        'idtrain': idtrain,
        'xtrain': xtrain,
        'ytrain': ytrain,
        'idval': idval,
        'xval': xval,
        'yval': yval,
        'submid': submid,
        'idtest': idtest,
        'xtest': xtest,
        'closed': closed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

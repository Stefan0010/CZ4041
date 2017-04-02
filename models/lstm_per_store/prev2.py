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
batch_size = {
    'train': 500,
    'val': 50,
    'test': 50
}
val_size = 100

# Load train & test data
data_dir = '../../data'

traind = util.load_train_data(data_dir)
testd = util.load_test_data(data_dir)

# Sort
traind.sort_values(['Date', 'Store'], ascending=True, inplace=True)
testd.sort_values(['Date', 'Store'], ascending=True, inplace=True)

# Drop all when stores are closed
traind = traind[traind['Open'] == 1]
days_open = testd['Open'] == 1

# Store
store_ids = testd['Store'].unique().astype(int, copy=False)
stores = {}

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

del traind['DayOfWeek']
del testd['DayOfWeek']

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  9, 18)
date_normer = float((date_max - date_min).days)

traind.insert(len(traind.columns), 'Date_Norm', (traind['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
testd.insert(len(testd.columns), 'Date_Norm', (testd['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)

# Sales
for store_id in store_ids:
    mask = traind['Store'] == store_id
    mean = traind.loc[mask, 'Sales'].mean()
    std = traind.loc[mask, 'Sales'].std()
    traind.loc[mask, 'Sales'] = (traind.loc[mask, 'Sales'] - mean) / std
    stores[store_id] = {
        'mean': mean,
        'std': std
    }

# Customers
del traind['Customers']

# Open

# Promo

# StateHoliday
del traind['StateHoliday']
del testd['StateHoliday']

# SchoolHoliday

# StateHoliday_0


del traind['StateHoliday_0']
del testd['StateHoliday_0']

# StateHoliday_a

del traind['StateHoliday_a']
del testd['StateHoliday_a']

# StateHoliday_b

del traind['StateHoliday_b']
del testd['StateHoliday_b']

# StateHoliday_c

del traind['StateHoliday_c']
del testd['StateHoliday_c']

# Weekends
del traind['Weekends']
del testd['Weekends']

# Weekdays
del traind['Weekdays']
del testd['Weekdays']

# StoreType_a
del traind['StoreType_a']
del testd['StoreType_a']

# StoreType_b
del traind['StoreType_b']
del testd['StoreType_b']

# StoreType_c
del traind['StoreType_c']
del testd['StoreType_c']

# StoreType_d
del traind['StoreType_d']
del testd['StoreType_d']

# Assortment_a
del traind['Assortment_a']
del testd['Assortment_a']

# Assortment_b
del traind['Assortment_b']
del testd['Assortment_b']

# Assortment_c
del traind['Assortment_c']
del testd['Assortment_c']

# HasCompetition
del traind['HasCompetition']
del testd['HasCompetition']

# CompetitionDistance
del traind['CompetitionDistance']
del testd['CompetitionDistance']

# max_compdist = traind['CompetitionDistance'].max()
# traind.loc[:, 'CompetitionDistance'] = traind['CompetitionDistance'] / max_compdist
# testd.loc[:, 'CompetitionDistance'] = testd['CompetitionDistance'] / max_compdist

# IsDoingPromo2

for store_id in store_ids:
    # Drops store id
    dfv = traind.loc[traind['Store'] == store_id]
    x = dfv.drop(['Store', 'Date', 'Sales', 'Open'], axis=1).as_matrix().astype(float, copy=False)
    y = dfv['Sales'].as_matrix().astype(float, copy=False)
    del dfv

    xtrain = x[:-val_size]
    ytrain = y[:-val_size]

    xval = x[-val_size:]
    yval = y[-val_size:]
    cont_val = np.ones(len(xval))
    cont_val[0] = 0
    assert len(xval) == len(yval)

    # Tweaks for batched learning
    assert len(xtrain) == len(ytrain)
    prev_len = len(xtrain)
    num_rem = batch_size['train'] - prev_len % batch_size['train']
    xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)
    ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)
    cont_train = np.ones(len(xtrain))
    cont_train[0] = 0
    cont_train[prev_len] = 0

    stores[store_id]['xtrain'] = xtrain
    stores[store_id]['ytrain'] = ytrain
    stores[store_id]['cont_train'] = cont_train

    stores[store_id]['xval'] = xval
    stores[store_id]['yval'] = yval
    stores[store_id]['cont_val'] = cont_val

    # Process its respective test data
    dfv = testd.loc[days_open & (testd['Store'] == store_id)]
    xtest = dfv.drop(['Id', 'Store', 'Date', 'Open'], axis=1).as_matrix().astype(float, copy=False)
    submid = dfv['Id'].as_matrix().astype(int, copy=False)
    del dfv

    # Tweak for batched learning
    assert len(xtest) == len(submid)
    prev_len = len(xtest)
    num_rem = batch_size['test'] - prev_len % batch_size['test']
    xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)
    cont_test = np.ones(len(xval))
    cont_test[0] = 0
    cont_test[prev_len] = 0

    stores[store_id]['xtest'] = xtest
    stores[store_id]['submid'] = submid
    stores[store_id]['cont_test'] = cont_test

zeroes = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

print 'xtrain.shape: ' + str(xtrain.shape)

with open('dsetv3.pickle', 'wb') as f:
    pickle.dump({
        'stores': stores,
        'zeroes': zeroes
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

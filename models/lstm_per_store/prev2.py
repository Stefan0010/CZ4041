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
batch_size = 50
val_size = 50

# Load train & test data
data_dir = '../../data'

traind = util.load_train_data(data_dir)
testd = util.load_test_data(data_dir)

# Sort
traind.sort_values(['Date', 'Store'], ascending=True, inplace=True)
testd.sort_values(['Date', 'Store'], ascending=True, inplace=True)

# Drop all when stores are closed
# traind = traind[traind['Open'] == 1]
days_open = testd['Open'] == 1

# Helper function
def mapper(val):
    if val == 'a':
        return 1
    elif val == 'b':
        return 2
    elif val == 'c':
        return 3
    else:
        return 0

# Customers
del traind['Customers']

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

# Open
# Promo
# SchoolHoliday
# IsDoingPromo2

# Store
store_ids = testd['Store'].unique().astype(int, copy=False)

stores = {}
for store_id in store_ids:
    stores[store_id] = {}
    mask = traind['Store'] == store_id

    x = traind.loc[mask].drop(['Store', 'DayOfWeek', 'Date', 'Sales', 'StateHoliday'], axis=1).as_matrix().astype(float, copy=False)
    y = traind.loc[mask, 'Sales'].as_matrix().astype(float, copy=False)

    xtrain = x[:-val_size]
    ytrain = y[:-val_size]
    prev_len = len(xtrain)
    num_rem = batch_size - prev_len % batch_size

    xval = x[-val_size:]
    yval = y[-val_size:]
    cont_val = np.ones(len(xval))
    cont_val[0] = 0

    mean = ytrain.mean()
    std = ytrain.std()
    ytrain = (ytrain - mean) / std
    yval = (yval - mean) / std

    stores[store_id]['mean'] = mean
    stores[store_id]['std'] = std

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

    # DayOfWeek
    dowvec = traind.loc[mask, 'DayOfWeek'].as_matrix().astype(int, copy=False) - 1

    dowtrain = dowvec[:-val_size]
    dowval = dowvec[-val_size:]

    dowtrain = np.insert(dowtrain, prev_len, dowtrain[:num_rem], 0)
    stores[store_id]['dowtrain'] = dowtrain
    stores[store_id]['dowval'] = dowval

    # Date
    dayvec = traind.loc[mask, 'Date'].dt.day.as_matrix().astype(int, copy=False) - 1
    monthvec = traind.loc[mask, 'Date'].dt.month.as_matrix().astype(int, copy=False) - 1
    yearvec = traind.loc[mask, 'Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

    daytrain = dayvec[:-val_size]
    monthtrain = monthvec[:-val_size]
    yeartrain = yearvec[:-val_size]

    dayval = dayvec[-val_size:]
    monthval = monthvec[-val_size:]
    yearval = yearvec[-val_size:]

    daytrain = np.insert(daytrain, prev_len, daytrain[:num_rem], 0)
    monthtrain = np.insert(monthtrain, prev_len, monthtrain[:num_rem], 0)
    yeartrain = np.insert(yeartrain, prev_len, yeartrain[:num_rem], 0)

    stores[store_id]['daytrain'] = daytrain
    stores[store_id]['monthtrain'] = monthtrain
    stores[store_id]['yeartrain'] = yeartrain

    stores[store_id]['dayval'] = dayval
    stores[store_id]['monthval'] = monthval
    stores[store_id]['yearval'] = yearval

    # StateHoliday
    sthvec = traind.loc[mask, 'StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)

    sthtrain = sthvec[:-val_size]
    sthval = sthvec[-val_size:]

    sthtrain = np.insert(sthtrain, prev_len, sthtrain[:num_rem], 0)

    stores[store_id]['sthtrain'] = sthtrain
    stores[store_id]['sthval'] = sthval

    #############################
    #  PROCESSING TEST DATASET  #
    #############################

    dfv = testd.loc[testd['Store'] == store_id]
    xtest = dfv.drop(['Id', 'Store', 'DayOfWeek', 'Date', 'StateHoliday'], axis=1).as_matrix().astype(float, copy=False)
    submid = dfv['Id'].as_matrix().astype(int, copy=False)
    del dfv

    # Tweak for batched learning
    prev_len = len(xtest)
    num_rem = batch_size - prev_len % batch_size
    xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)
    cont_test = np.ones(len(xval))
    cont_test[0] = 0
    cont_test[prev_len] = 0

    stores[store_id]['xtest'] = xtest
    stores[store_id]['submid'] = submid
    stores[store_id]['cont_test'] = cont_test

    # DayOfWeek
    dowtest = testd.loc[mask, 'DayOfWeek'].as_matrix().astype(int, copy=False) - 1

    prev_len = len(dowtest)
    num_rem = batch_size - prev_len % batch_size

    dowtest = np.insert(dowtest, prev_len, dowtest[:num_rem], 0)
    stores[store_id]['dowtest'] = dowtest

    # Date
    daytest = testd.loc[mask, 'Date'].dt.day.as_matrix().astype(int, copy=False) - 1
    monthtest = testd.loc[mask, 'Date'].dt.month.as_matrix().astype(int, copy=False) - 1
    yeartest = testd.loc[mask, 'Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

    daytest = np.insert(daytest, prev_len, daytest[:num_rem], 0)
    monthtest = np.insert(monthtest, prev_len, monthtest[:num_rem], 0)
    yeartest = np.insert(yeartest, prev_len, yeartest[:num_rem], 0)

    stores[store_id]['daytest'] = daytest
    stores[store_id]['monthtest'] = monthtest
    stores[store_id]['yeartest'] = yeartest

    # StateHoliday
    sthtest = testd.loc[mask, 'StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)

    sthtest = np.insert(sthtest, prev_len, sthtest[:num_rem], 0)

    stores[store_id]['sthtest'] = sthtest

zeroes = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

print 'xtrain.shape: ' + str(xtrain.shape)

with open('dsetv3.pickle', 'wb') as f:
    pickle.dump({
        'stores': stores,
        'zeroes': zeroes
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

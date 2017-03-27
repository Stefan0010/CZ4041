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
batch_size = 100000

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

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  9, 18)
date_normer = float((date_max - date_min).days)

traind.insert(len(traind.columns), 'Date_Norm', (traind['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
traind.insert(len(traind.columns), 'Day_Norm', (traind['Date'].dt.day - 1.) / 31.)
traind.insert(len(traind.columns), 'Month_Norm', (traind['Date'].dt.month - 1.) / 12.)
traind.insert(len(traind.columns), 'Year_Norm', (traind['Date'].dt.year - 2013.) / 2.)

testd.insert(len(testd.columns), 'Date_Norm', (testd['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
testd.insert(len(testd.columns), 'Day_Norm', (testd['Date'].dt.day - 1.) / 31.)
testd.insert(len(testd.columns), 'Month_Norm', (testd['Date'].dt.month - 1.) / 12.)
testd.insert(len(testd.columns), 'Year_Norm', (testd['Date'].dt.year - 2013.) / 2.)

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

std = traind['Sales'].std()
mean = traind['Sales'].mean()
traind.loc[:, 'Sales'] = (traind['Sales'] - mean) / std

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

traind.loc[traind['NoCompetition'], 'CompetitionDistance'] = 0.
testd.loc[testd['NoCompetition'], 'CompetitionDistance'] = 0.

# IsDoingPromo2

xtrain = traind.drop(['Store', 'DayOfWeek', 'Date', 'Sales', 'Open'], axis=1).as_matrix().astype(float, copy=False)
ytrain = traind['Sales'].as_matrix().astype(float, copy=False)

prev_len = len(xtrain)
num_rem = batch_size - prev_len % batch_size
xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)
ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)

submid = testd.loc[testd['Open'] == 1, 'Id'].as_matrix().astype(int, copy=False)
xtest = testd.loc[testd['Open'] == 1].drop(['Id', 'Store', 'DayOfWeek', 'Date', 'Open'], axis=1).as_matrix().astype(float, copy=False)
closed = testd.loc[testd['Open'] == 0, 'Id'].as_matrix().astype(int, copy=False)

prev_len = len(xtest)

# Number of test samples << batch_size
num_rem = batch_size - prev_len % batch_size
for i in range(0, num_rem, prev_len):
    xtest = np.insert(xtest, prev_len, xtest[: min(prev_len, batch_size - len(xtest)) ], 0)

with open('dset.pickle', 'wb') as f:
    pickle.dump({
        'xtrain': xtrain,
        'ytrain': ytrain,
        'submid': submid,
        'xtest': xtest,
        'closed': closed,
        'mean': mean,
        'std': std
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

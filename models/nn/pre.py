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
batch_size = 5000
val_size = 10000

# Load train & test data
data_dir = '../../data'

traind = util.load_train_data(data_dir)
testd = util.load_test_data(data_dir)

def znorm(df1, df2, column):
    mean = df1[column].mean()
    std = df1[column].std()
    df1.loc[:, column] = (df1[column] - mean) / std
    df2.loc[:, column] = (df2[column] - mean) / std

# Sort
traind.sort_values(['Date', 'Store'], ascending=True, inplace=True)
testd.sort_values(['Date', 'Store'], ascending=True, inplace=True)

# Drop all when stores are closed
# traind = traind[traind['Open'] == 1]
days_open = testd['Open'] == 1

# Store
store_ids = traind['Store'].unique().astype(int, copy=False)

idvec = traind['Store'].as_matrix().astype(int, copy=False) - 1
idtest = testd.loc[days_open, 'Store'].as_matrix().astype(int, copy=False) - 1

# DayOfWeek
dowvec = traind['DayOfWeek'].as_matrix().astype(int, copy=False) - 1
dow_test = testd.loc[days_open, 'DayOfWeek'].as_matrix().astype(int, copy=False) - 1

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  7, 31)

dayvec = traind['Date'].dt.day.as_matrix().astype(int, copy=False) - 1
monthvec = traind['Date'].dt.month.as_matrix().astype(int, copy=False) - 1
yearvec = traind['Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

day_test = testd.loc[days_open, 'Date'].dt.day.as_matrix().astype(int, copy=False) - 1
month_test = testd.loc[days_open, 'Date'].dt.month.as_matrix().astype(int, copy=False) - 1
year_test = testd.loc[days_open, 'Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

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
def mapper(val):
    if val == 'a':
        return 1
    elif val == 'b':
        return 2
    elif val == 'c':
        return 3
    else:
        return 0

sthvec = traind['StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)
sth_test = testd.loc[days_open, 'StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)

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
stypevec = np.zeros(len(traind))
stypevec[traind['StoreType_b'].as_matrix() == 1] = 1
stypevec[traind['StoreType_c'].as_matrix() == 1] = 2
stypevec[traind['StoreType_d'].as_matrix() == 1] = 3

stype_test = np.zeros(days_open.sum())
stype_test[testd.loc[days_open, 'StoreType_b'].as_matrix() == 1] = 1
stype_test[testd.loc[days_open, 'StoreType_c'].as_matrix() == 1] = 2
stype_test[testd.loc[days_open, 'StoreType_d'].as_matrix() == 1] = 3

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
atypevec = np.zeros(len(traind))
atypevec[traind['Assortment_b'].as_matrix() == 1] = 1
atypevec[traind['Assortment_c'].as_matrix() == 1] = 2

atype_test = np.zeros(days_open.sum())
atype_test[testd.loc[days_open, 'Assortment_b'].as_matrix() == 1] = 1
atype_test[testd.loc[days_open, 'Assortment_c'].as_matrix() == 1] = 2

del traind['Assortment_a']
del testd['Assortment_a']

# Assortment_b
del traind['Assortment_b']
del testd['Assortment_b']

# Assortment_c
del traind['Assortment_c']
del testd['Assortment_c']

# HasCompetition
traind.insert(len(traind.columns), 'NoCompetition', traind['HasCompetition'] == 0)
testd.insert(len(testd.columns), 'NoCompetition', testd['HasCompetition'] == 0)

# CompetitionDistance
max_compdist = traind['CompetitionDistance'].max()
traind.loc[:, 'CompetitionDistance'] = traind['CompetitionDistance'] / max_compdist
testd.loc[:, 'CompetitionDistance'] = testd['CompetitionDistance'] / max_compdist

traind.loc[traind['NoCompetition'], 'CompetitionDistance'] = 0.0
testd.loc[testd['NoCompetition'], 'CompetitionDistance'] = 0.0

# IsDoingPromo2


x = traind.drop(['Store', 'DayOfWeek', 'Date', 'Sales'], axis=1).as_matrix().astype(float, copy=False)
y = traind['Sales'].as_matrix().astype(float, copy=False)

shuffledidx = np.arange(len(idvec))
np.random.shuffle(shuffledidx)
idvec = idvec[shuffledidx]
dayvec = dayvec[shuffledidx]
monthvec = monthvec[shuffledidx]
yearvec = yearvec[shuffledidx]
dowvec = dowvec[shuffledidx]
sthvec = sthvec[shuffledidx]
stypevec = stypevec[shuffledidx]
atypevec = atypevec[shuffledidx]
x = x[shuffledidx]
y = y[shuffledidx]

idtrain = idvec[:-val_size]
day_train = dayvec[:-val_size]
month_train = monthvec[:-val_size]
year_train = yearvec[:-val_size]
dow_train = dowvec[:-val_size]
sth_train = sthvec[:-val_size]
stype_train = stypevec[:-val_size]
atype_train = atypevec[:-val_size]
xtrain = x[:-val_size]
ytrain = y[:-val_size]

idval = idvec[-val_size:]
day_val = dayvec[-val_size:]
month_val = monthvec[-val_size:]
year_val = yearvec[-val_size:]
dow_val = dowvec[-val_size:]
sth_val = sthvec[-val_size:]
stype_val = stypevec[-val_size:]
atype_val = atypevec[-val_size:]
xval = x[-val_size:]
yval = y[-val_size:]

prev_len = len(xtrain)
num_rem = batch_size - prev_len % batch_size
idtrain = np.insert(idtrain, prev_len, idtrain[:num_rem], 0)
day_train = np.insert(day_train, prev_len, day_train[:num_rem], 0)
month_train = np.insert(month_train, prev_len, month_train[:num_rem], 0)
year_train = np.insert(year_train, prev_len, year_train[:num_rem], 0)
dow_train = np.insert(dow_train, prev_len, dow_train[:num_rem], 0)
sth_train = np.insert(sth_train, prev_len, sth_train[:num_rem], 0)
stype_train = np.insert(stype_train, prev_len, stype_train[:num_rem], 0)
atype_train = np.insert(atype_train, prev_len, atype_train[:num_rem], 0)
xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)
ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)

submid = testd.loc[days_open, 'Id'].as_matrix().astype(int, copy=False)
xtest = testd.loc[days_open].drop(['Id', 'Store', 'DayOfWeek', 'Date'], axis=1).as_matrix().astype(float, copy=False)
closed = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

prev_len = len(xtest)
num_rem = batch_size - prev_len % batch_size
idtest = np.insert(idtest, prev_len, idtest[:num_rem], 0)
day_test = np.insert(day_test, prev_len, day_test[:num_rem], 0)
month_test = np.insert(month_test, prev_len, month_test[:num_rem], 0)
year_test = np.insert(year_test, prev_len, year_test[:num_rem], 0)
dow_test = np.insert(dow_test, prev_len, dow_test[:num_rem], 0)
sth_test = np.insert(sth_test, prev_len, sth_test[:num_rem], 0)
stype_test = np.insert(stype_test, prev_len, stype_test[:num_rem], 0)
atype_test = np.insert(atype_test, prev_len, atype_test[:num_rem], 0)
xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)

with open('dset.pickle', 'wb') as f:
    pickle.dump({
        'mean': mean,
        'std': std,
        'idtrain': idtrain,
        'day_train': day_train,
        'month_train': month_train,
        'year_train': year_train,
        'dow_train': dow_train,
        'sth_train': sth_train,
        'stype_train': stype_train,
        'atype_train': atype_train,
        'xtrain': xtrain,
        'ytrain': ytrain,
        'idval': idval,
        'day_val': day_val,
        'month_val': month_val,
        'year_val': year_val,
        'dow_val': dow_val,
        'sth_val': sth_val,
        'stype_val': stype_val,
        'atype_val': atype_val,
        'xval': xval,
        'yval': yval,
        'submid': submid,
        'idtest': idtest,
        'day_test': day_test,
        'month_test': month_test,
        'year_test': year_test,
        'dow_test': dow_test,
        'sth_test': sth_test,
        'stype_test': stype_test,
        'atype_test': atype_test,
        'xtest': xtest,
        'closed': closed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

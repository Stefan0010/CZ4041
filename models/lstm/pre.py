import os
import sys

import numpy as np
import pandas as pd

import pickle
from datetime import datetime

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

print 'Loading training, and test dataset'

data_dir = '../../data'
traind = util.load_train_data(data_dir)
testd = util.load_test_data(data_dir)

# We do not concern with closed days
# traind = traind[traind['Open'] == 1]
days_open = testd['Open'] == 1

print 'Dataset loaded'

traind.sort_values(['Store', 'Date'], ascending=True, inplace=True)
testd.sort_values(['Store', 'Date'], ascending=True, inplace=True)

batch_size = 5000

# Slice validation set
num_val_stores = 50
shuffledids = np.arange(1, 1116)
np.random.shuffle(shuffledids)
idval = shuffledids[:num_val_stores]

val_mask = traind['Store'] == idval[0]
for i in range(1, num_val_stores):
    val_mask = val_mask | (traind['Store'] == idval[i])

vald = traind.loc[val_mask]
traind = traind.loc[~val_mask]

# Helper function
def znorm(df1, df2, column):
    mean = df1[column].mean()
    std = df1[column].std()
    df1.loc[:, column] = (df1[column] - mean) / std
    df2.loc[:, column] = (df2[column] - mean) / std

print 'Starting preprocessing'

# Store
# Shift from 1..1115 to 0..1114
traind.loc[:, 'Store'] -= 1
vald.loc[:, 'Store'] -= 1
testd.loc[:, 'Store'] -= 1

store_ids = traind['Store'].unique().astype(int, copy=False)

idtrain = traind['Store'].as_matrix().astype(int, copy=False)
idval = vald['Store'].as_matrix().astype(int, copy=False)
idtest = testd.loc[days_open, 'Store'].as_matrix().astype(int, copy=False)

# DayOfWeek
traind.loc[:, 'DayOfWeek'] -= 1
vald.loc[:, 'DayOfWeek'] -= 1
testd.loc[:, 'DayOfWeek'] -= 1

dowtrain = traind['DayOfWeek'].as_matrix().astype(int, copy=False)
dowval = vald['DayOfWeek'].as_matrix().astype(int, copy=False)
dowtest = testd.loc[days_open, 'DayOfWeek'].as_matrix().astype(int, copy=False)

# Date
daytrain = traind['Date'].dt.day.as_matrix().astype(int, copy=False) - 1
monthtrain = traind['Date'].dt.month.as_matrix().astype(int, copy=False) - 1
yeartrain = traind['Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

dayval = vald['Date'].dt.day.as_matrix().astype(int, copy=False) - 1
monthval = vald['Date'].dt.month.as_matrix().astype(int, copy=False) - 1
yearval = vald['Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

daytest = testd.loc[days_open, 'Date'].dt.day.as_matrix().astype(int, copy=False) - 1
monthtest = testd.loc[days_open, 'Date'].dt.month.as_matrix().astype(int, copy=False) - 1
yeartest = testd.loc[days_open, 'Date'].dt.year.as_matrix().astype(int, copy=False) - 2013

# Sales
mean = traind['Sales'].mean()
std = traind['Sales'].std()

traind.loc[:, 'Sales'] = (traind['Sales'] - mean) / std
vald.loc[:, 'Sales'] = (vald['Sales'] - mean) / std

# Customers
del traind['Customers']
del vald['Customers']

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

sthtrain = traind['StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)
sthval = vald['StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)
sthtest = testd.loc[days_open, 'StateHoliday'].map(mapper).as_matrix().astype(int, copy=False)

del traind['StateHoliday']
del vald['StateHoliday']
del testd['StateHoliday']

# SchoolHoliday

# StateHoliday_0
del traind['StateHoliday_0']
del vald['StateHoliday_0']
del testd['StateHoliday_0']

# StateHoliday_a
del traind['StateHoliday_a']
del vald['StateHoliday_a']
del testd['StateHoliday_a']

# StateHoliday_b
del traind['StateHoliday_b']
del vald['StateHoliday_b']
del testd['StateHoliday_b']

# StateHoliday_c
del traind['StateHoliday_c']
del vald['StateHoliday_c']
del testd['StateHoliday_c']

# Weekends
del traind['Weekends']
del vald['Weekends']
del testd['Weekends']

# Weekdays
del traind['Weekdays']
del vald['Weekdays']
del testd['Weekdays']

# StoreType_a
stypetrain = np.zeros(len(traind))
stypetrain[traind['StoreType_b'].as_matrix() == 1] = 1
stypetrain[traind['StoreType_c'].as_matrix() == 1] = 2
stypetrain[traind['StoreType_d'].as_matrix() == 1] = 3

stypeval = np.zeros(len(vald))
stypeval[vald['StoreType_b'].as_matrix() == 1] = 1
stypeval[vald['StoreType_c'].as_matrix() == 1] = 2
stypeval[vald['StoreType_d'].as_matrix() == 1] = 3

stypetest = np.zeros(days_open.sum())
stypetest[testd.loc[days_open, 'StoreType_b'].as_matrix() == 1] = 1
stypetest[testd.loc[days_open, 'StoreType_c'].as_matrix() == 1] = 2
stypetest[testd.loc[days_open, 'StoreType_d'].as_matrix() == 1] = 3

del traind['StoreType_a']
del vald['StoreType_a']
del testd['StoreType_a']

# StoreType_b
del traind['StoreType_b']
del vald['StoreType_b']
del testd['StoreType_b']

# StoreType_c
del traind['StoreType_c']
del vald['StoreType_c']
del testd['StoreType_c']

# StoreType_d
del traind['StoreType_d']
del vald['StoreType_d']
del testd['StoreType_d']

# Assortment_a
atypetrain = np.zeros(len(traind))
atypetrain[traind['Assortment_b'].as_matrix() == 1] = 1
atypetrain[traind['Assortment_c'].as_matrix() == 1] = 2

atypeval = np.zeros(len(vald))
atypeval[vald['Assortment_b'].as_matrix() == 1] = 1
atypeval[vald['Assortment_c'].as_matrix() == 1] = 2

atypetest = np.zeros(days_open.sum())
atypetest[testd.loc[days_open, 'Assortment_b'].as_matrix() == 1] = 1
atypetest[testd.loc[days_open, 'Assortment_c'].as_matrix() == 1] = 2

del traind['Assortment_a']
del vald['Assortment_a']
del testd['Assortment_a']

# Assortment_b
del traind['Assortment_b']
del vald['Assortment_b']
del testd['Assortment_b']

# Assortment_c
del traind['Assortment_c']
del vald['Assortment_c']
del testd['Assortment_c']

# HasCompetition

# CompetitionDistance
max_compdist = traind['CompetitionDistance'].max()
traind.loc[:, 'CompetitionDistance'] = traind['CompetitionDistance'] / max_compdist
vald.loc[:, 'CompetitionDistance'] = vald['CompetitionDistance'] / max_compdist
testd.loc[:, 'CompetitionDistance'] = testd['CompetitionDistance'] / max_compdist

traind.loc[traind['HasCompetition'] == 0, 'CompetitionDistance'] = 5.0
vald.loc[vald['HasCompetition'] == 0, 'CompetitionDistance'] = 5.0
testd.loc[testd['HasCompetition'] == 0, 'CompetitionDistance'] = 5.0

# IsDoingPromo2

xtrain = traind.drop(['Store', 'DayOfWeek', 'Date', 'Sales'], axis=1).as_matrix().astype(float, copy=False)
ytrain = traind['Sales'].as_matrix().astype(float, copy=False)

xval = vald.drop(['Store', 'DayOfWeek', 'Date', 'Sales'], axis=1).as_matrix().astype(float, copy=False)
yval = vald['Sales'].as_matrix().astype(float, copy=False)

submid = testd.loc[days_open, 'Id'].as_matrix().astype(int, copy=False)
closed = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)
xtest = testd.loc[days_open].drop(['Id', 'Store', 'DayOfWeek', 'Date'], axis=1).as_matrix().astype(float, copy=False)

assert len(xtrain) == len(idtrain) == len(daytrain) == len(monthtrain) == len(yeartrain) == len(dowtrain)
assert len(xtrain) == len(sthtrain) == len(stypetrain) == len(atypetrain)

assert len(xval) == len(idval) == len(dayval) == len(monthval) == len(yearval) == len(dowval)
assert len(xval) == len(sthval) == len(stypeval) == len(atypeval)

assert len(xtest) == len(idtest) == len(daytest) == len(monthtest) == len(yeartest) == len(dowtest)
assert len(xtest) == len(sthtest) == len(stypetest) == len(atypetest)

prev_len = len(xtrain)
num_rem = batch_size - prev_len % batch_size
idtrain = np.insert(idtrain, prev_len, idtrain[:num_rem], 0)
daytrain = np.insert(daytrain, prev_len, daytrain[:num_rem], 0)
monthtrain = np.insert(monthtrain, prev_len, monthtrain[:num_rem], 0)
yeartrain = np.insert(yeartrain, prev_len, yeartrain[:num_rem], 0)
dowtrain = np.insert(dowtrain, prev_len, dowtrain[:num_rem], 0)
sthtrain = np.insert(sthtrain, prev_len, sthtrain[:num_rem], 0)
stypetrain = np.insert(stypetrain, prev_len, stypetrain[:num_rem], 0)
atypetrain = np.insert(atypetrain, prev_len, atypetrain[:num_rem], 0)
xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)
ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)

cont_train = np.zeros(len(xtrain))
for i in range(1, len(xtrain)):
    if idtrain[i] == idtrain[i - 1]:
        cont_train[i] = 1

prev_len = len(xval)
num_rem = batch_size - prev_len % batch_size
idval = np.insert(idval, prev_len, idval[:num_rem], 0)
dayval = np.insert(dayval, prev_len, dayval[:num_rem], 0)
monthval = np.insert(monthval, prev_len, monthval[:num_rem], 0)
yearval = np.insert(yearval, prev_len, yearval[:num_rem], 0)
dowval = np.insert(dowval, prev_len, dowval[:num_rem], 0)
sthval = np.insert(sthval, prev_len, sthval[:num_rem], 0)
stypeval = np.insert(stypeval, prev_len, stypeval[:num_rem], 0)
atypeval = np.insert(atypeval, prev_len, atypeval[:num_rem], 0)
xval = np.insert(xval, prev_len, xval[:num_rem], 0)
yval = np.insert(yval, prev_len, yval[:num_rem], 0)

cont_val = np.zeros(len(xval))
for i in range(1, len(xval)):
    if idval[i] == idval[i - 1]:
        cont_val[i] = 1

prev_len = len(xtest)
num_rem = batch_size - prev_len % batch_size
idtest = np.insert(idtest, prev_len, idtest[:num_rem], 0)
daytest = np.insert(daytest, prev_len, daytest[:num_rem], 0)
monthtest = np.insert(monthtest, prev_len, monthtest[:num_rem], 0)
yeartest = np.insert(yeartest, prev_len, yeartest[:num_rem], 0)
dowtest = np.insert(dowtest, prev_len, dowtest[:num_rem], 0)
sthtest = np.insert(sthtest, prev_len, sthtest[:num_rem], 0)
stypetest = np.insert(stypetest, prev_len, stypetest[:num_rem], 0)
atypetest = np.insert(atypetest, prev_len, atypetest[:num_rem], 0)
xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)

cont_test = np.zeros(len(xtest))
for i in range(1, len(xtest)):
    if idtest[i] == idtest[i - 1]:
        cont_test[i] = 1

with open('dset.pickle', 'wb') as f:
    pickle.dump({
        'mean': mean,
        'std': std,

        'xtrain': xtrain,
        'cont_train': cont_train,
        'ytrain': ytrain,
        'idtrain': idtrain,
        'daytrain': daytrain,
        'monthtrain': monthtrain,
        'yeartrain': yeartrain,
        'dowtrain': dowtrain,
        'sthtrain': sthtrain,
        'stypetrain': stypetrain,
        'atypetrain': atypetrain,

        'xval': xval,
        'cont_val': cont_val,
        'yval': yval,
        'idval': idval,
        'dayval': dayval,
        'monthval': monthval,
        'yearval': yearval,
        'dowval': dowval,
        'sthval': sthval,
        'stypeval': stypeval,
        'atypeval': atypeval,

        'submid': submid,
        'xtest': xtest,
        'cont_test': cont_test,
        'idtest': idtest,
        'daytest': daytest,
        'monthtest': monthtest,
        'yeartest': yeartest,
        'dowtest': dowtest,
        'sthtest': sthtest,
        'stypetest': stypetest,
        'atypetest': atypetest,
        'closed': closed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

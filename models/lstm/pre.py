import os
import sys

import numpy as np
import pandas as pd

import pickle
from datetime import datetime

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

print 'Loading training, validation, and test dataset'

data_dir = '../data'
traind, vald = util.load_splitted_data(data_dir)
testd = util.load_test_data(data_dir)

# We do not concern with closed days
traind = traind[traind['Open'] == 1]
vald = vald[vald['Open'] == 1]
testd = testd[testd['Open'] == 1]

print 'Dataset loaded'

traind.sort_values(['Store', 'Date'], ascending=True, inplace=True)
vald.sort_values(['Store', 'Date'], ascending=True, inplace=True)
testd.sort_values(['Store', 'Date'], ascending=True, inplace=True)

batch_size = 10000

stores = {}

train_ids = traind['Store'].unique().astype(int, copy=False)
val_ids = vald['Store'].unique().astype(int, copy=False)
test_ids = testd['Store'].unique().astype(int, copy=False)

print 'Starting preprocessing'

# Store

# DayOfWeek
traind.loc[:, 'DayOfWeek'] = (traind['DayOfWeek'] - 1.) / 6.
vald.loc[:, 'DayOfWeek'] = (vald['DayOfWeek'] - 1.) / 6.
testd.loc[:, 'DayOfWeek'] = (testd['DayOfWeek'] - 1.) / 6.

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  9, 17)
date_norm = float((date_max - date_min).days)

traind.insert(len(traind.columns), 'Date_Norm', (traind['Date'] - date_min) / np.timedelta64(1, 'D') / date_norm)
vald.insert(len(vald.columns), 'Date_Norm', (vald['Date'] - date_min) / np.timedelta64(1, 'D') / date_norm)
testd.insert(len(testd.columns), 'Date_Norm', (testd['Date'] - date_min) / np.timedelta64(1, 'D') / date_norm)

# Sales
for sid in train_ids:
    mask = traind['Store'] == sid
    mean = traind.loc[mask, 'Sales'].mean()
    std = traind.loc[mask, 'Sales'].std()
    stores[sid] = {
        'mean': mean,
        'std': std
    }

    # Normalize
    traind.loc[mask, 'Sales'] = (traind.loc[mask, 'Sales'] - mean) / std

for sid in val_ids:
    mask = vald['Store'] == sid
    mean = vald.loc[mask, 'Sales'].mean()
    std = vald.loc[mask, 'Sales'].std()
    stores[sid] = {
        'mean': mean,
        'std': std
    }

    # Normalize
    vald.loc[mask, 'Sales'] = (vald.loc[mask, 'Sales'] - mean) / std

with open('stores.pickle', 'wb') as f:
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)

# Customers
del traind['Customers']
del vald['Customers']

# Open
del traind['Open']
del vald['Open']
del testd['Open']

# Promo

# StateHoliday
del traind['StateHoliday']
del vald['StateHoliday']
del testd['StateHoliday']

# SchoolHoliday

# StateHoliday_0

# StateHoliday_a

# StateHoliday_b

# StateHoliday_c

# Weekends
del traind['Weekends']
del vald['Weekends']
del testd['Weekends']

# Weekdays
del traind['Weekdays']
del vald['Weekdays']
del testd['Weekdays']

# StoreType_a

# StoreType_b

# StoreType_c

# StoreType_d

# Assortment_a

# Assortment_b

# Assortment_c

# HasCompetition

# CompetitionDistance
max_comp_dist = max(traind['CompetitionDistance'].max(), vald['CompetitionDistance'].max())

traind.loc[traind['HasCompetition'] == 0, 'CompetitionDistance'] = max_comp_dist
vald.loc[vald['HasCompetition'] == 0, 'CompetitionDistance'] = max_comp_dist
testd.loc[testd['HasCompetition'] == 0, 'CompetitionDistance'] = max_comp_dist

traind.loc[:, 'CompetitionDistance'] /= max_comp_dist
vald.loc[:, 'CompetitionDistance'] /= max_comp_dist
testd.loc[:, 'CompetitionDistance'] /= max_comp_dist

# IsDoingPromo2

print 'Preprocessing 50%% done'

idtrain = traind['Store'].as_matrix()
xtrain = traind.drop(['Store', 'Sales', 'Date'], axis=1).as_matrix().astype(float, copy=False)
ytrain = traind['Sales'].as_matrix().astype(float, copy=False)
cont_train = np.ones(len(xtrain))

cont_train[0] = 0
for i in range(1, len(xtrain)):
    if traind['Store'].iloc[i] != traind['Store'].iloc[i - 1]:
        cont_train[i] = 0

assert len(idtrain) == len(xtrain) == len(ytrain) == len(cont_train)
# Fix to match batch size
prev_len = len(xtrain)
num_rem = batch_size - prev_len % batch_size
idtrain = np.insert(idtrain, prev_len, idtrain[:num_rem], 0)
xtrain = np.insert(xtrain, prev_len, xtrain[:num_rem], 0)

print 'Preprocessing 62.5%% done'

ytrain = np.insert(ytrain, prev_len, ytrain[:num_rem], 0)
cont_train = np.insert(cont_train, prev_len, cont_train[:num_rem], 0)

# del traind

with open('traind.pickle', 'wb') as f:
    pickle.dump({
        'idtrain': idtrain,
        'xtrain': xtrain,
        'ytrain': ytrain,
        'cont_train': cont_train,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

print 'Preprocessing 75%% done'

idval = vald['Store'].as_matrix()
xval = vald.drop(['Store', 'Sales', 'Date'], axis=1).as_matrix().astype(float, copy=False)
yval = vald['Sales'].as_matrix().astype(float, copy=False)
cont_val = np.ones(len(xval))

cont_val[0] = 0
for i in range(1, len(xval)):
    if vald['Store'].iloc[i] != vald['Store'].iloc[i - 1]:
        cont_val[i] = 0

assert len(idval) == len(xval) == len(yval) == len(cont_val)
# Fix to match batch size
prev_len = len(xval)
num_rem = batch_size - prev_len % batch_size
idval = np.insert(idval, prev_len, idval[:num_rem], 0)
xval = np.insert(xval, prev_len, xval[:num_rem], 0)

print 'Preprocessing 87.5%% done'

yval = np.insert(yval, prev_len, yval[:num_rem], 0)
cont_val = np.insert(cont_val, prev_len, cont_val[:num_rem], 0)

# del vald

with open('vald.pickle', 'wb') as f:
    pickle.dump({
        'idval': idval,
        'xval': xval,
        'yval': yval,
        'cont_val': cont_val,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

print 'Preprocessing done!\nPreprocessing test dataset'

submid = testd['Id'].as_matrix()
idtest = testd['Store'].as_matrix()
xtest = testd.drop(['Id', 'Store', 'Date'], axis=1).as_matrix().astype(float, copy=False)
cont_test = np.ones(len(xtest))

cont_test[0] = 0
for i in range(1, len(xtest)):
    if testd['Store'].iloc[i] != testd['Store'].iloc[i - 1]:
        cont_test[i] = 0

assert len(submid) == len(idtest) == len(xtest) == len(cont_test)
prev_len = len(xtest)
num_rem = batch_size - prev_len % batch_size
idtest = np.insert(idtest, prev_len, idtest[:num_rem], 0)
xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)

print 'Preprocessing test dataset 50%% done'

submid = np.insert(submid, prev_len, submid[:num_rem], 0)
cont_test = np.insert(cont_test, prev_len, cont_test[:num_rem], 0)

with open('testd.pickle', 'wb') as f:
    pickle.dump({
        'idtest': idtest,
        'xtest': xtest,
        'submid': submid,
        'cont_test': cont_test
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

print 'Preprocessing test dataset done!'

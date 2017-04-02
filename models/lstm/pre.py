import os
import sys

import numpy as np
import pandas as pd

import pickle
from datetime import datetime

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

print 'Loading training, validation, and test dataset'

data_dir = '../../data'
traind, vald = util.load_splitted_data(data_dir)
testd = util.load_test_data(data_dir)

# We do not concern with closed days
traind = traind[traind['Open'] == 1]
vald = vald[vald['Open'] == 1]
days_open = testd['Open'] == 1

print 'Dataset loaded'

traind.sort_values(['Store', 'Date'], ascending=True, inplace=True)
vald.sort_values(['Store', 'Date'], ascending=True, inplace=True)
testd.sort_values(['Store', 'Date'], ascending=True, inplace=True)

batch_size = 5000

stores = {}

train_ids = traind['Store'].unique().astype(int, copy=False)
val_ids = vald['Store'].unique().astype(int, copy=False)
test_ids = testd.loc[days_open, 'Store'].unique().astype(int, copy=False)

print 'Starting preprocessing'

# Store
traind.insert(len(traind.columns), 'Store_0', 0)
traind.insert(len(traind.columns), 'Store_1', 0)
traind.insert(len(traind.columns), 'Store_2', 0)
traind.insert(len(traind.columns), 'Store_3', 0)
traind.insert(len(traind.columns), 'Store_4', 0)
traind.insert(len(traind.columns), 'Store_5', 0)
traind.insert(len(traind.columns), 'Store_6', 0)
traind.insert(len(traind.columns), 'Store_7', 0)
traind.insert(len(traind.columns), 'Store_8', 0)
traind.insert(len(traind.columns), 'Store_9', 0)
traind.insert(len(traind.columns), 'Store_10', 0)

vald.insert(len(vald.columns), 'Store_0', 0)
vald.insert(len(vald.columns), 'Store_1', 0)
vald.insert(len(vald.columns), 'Store_2', 0)
vald.insert(len(vald.columns), 'Store_3', 0)
vald.insert(len(vald.columns), 'Store_4', 0)
vald.insert(len(vald.columns), 'Store_5', 0)
vald.insert(len(vald.columns), 'Store_6', 0)
vald.insert(len(vald.columns), 'Store_7', 0)
vald.insert(len(vald.columns), 'Store_8', 0)
vald.insert(len(vald.columns), 'Store_9', 0)
vald.insert(len(vald.columns), 'Store_10', 0)

testd.insert(len(testd.columns), 'Store_0', 0)
testd.insert(len(testd.columns), 'Store_1', 0)
testd.insert(len(testd.columns), 'Store_2', 0)
testd.insert(len(testd.columns), 'Store_3', 0)
testd.insert(len(testd.columns), 'Store_4', 0)
testd.insert(len(testd.columns), 'Store_5', 0)
testd.insert(len(testd.columns), 'Store_6', 0)
testd.insert(len(testd.columns), 'Store_7', 0)
testd.insert(len(testd.columns), 'Store_8', 0)
testd.insert(len(testd.columns), 'Store_9', 0)
testd.insert(len(testd.columns), 'Store_10', 0)

for store_id in range(1, 1116):
    bitarr = [store_id >> i & 1 for i in range(0, 11)]

    mask = traind['Store'] == store_id
    traind.loc[mask, 'Store_0'] = bitarr[0]
    traind.loc[mask, 'Store_1'] = bitarr[1]
    traind.loc[mask, 'Store_2'] = bitarr[2]
    traind.loc[mask, 'Store_3'] = bitarr[3]
    traind.loc[mask, 'Store_4'] = bitarr[4]
    traind.loc[mask, 'Store_5'] = bitarr[5]
    traind.loc[mask, 'Store_6'] = bitarr[6]
    traind.loc[mask, 'Store_7'] = bitarr[7]
    traind.loc[mask, 'Store_8'] = bitarr[8]
    traind.loc[mask, 'Store_9'] = bitarr[9]
    traind.loc[mask, 'Store_10'] = bitarr[10]

    mask2 = vald['Store'] == store_id
    vald.loc[mask2, 'Store_0'] = bitarr[0]
    vald.loc[mask2, 'Store_1'] = bitarr[1]
    vald.loc[mask2, 'Store_2'] = bitarr[2]
    vald.loc[mask2, 'Store_3'] = bitarr[3]
    vald.loc[mask2, 'Store_4'] = bitarr[4]
    vald.loc[mask2, 'Store_5'] = bitarr[5]
    vald.loc[mask2, 'Store_6'] = bitarr[6]
    vald.loc[mask2, 'Store_7'] = bitarr[7]
    vald.loc[mask2, 'Store_8'] = bitarr[8]
    vald.loc[mask2, 'Store_9'] = bitarr[9]
    vald.loc[mask2, 'Store_10'] = bitarr[10]

    mask3 = testd['Store'] == store_id
    testd.loc[mask3, 'Store_0'] = bitarr[0]
    testd.loc[mask3, 'Store_1'] = bitarr[1]
    testd.loc[mask3, 'Store_2'] = bitarr[2]
    testd.loc[mask3, 'Store_3'] = bitarr[3]
    testd.loc[mask3, 'Store_4'] = bitarr[4]
    testd.loc[mask3, 'Store_5'] = bitarr[5]
    testd.loc[mask3, 'Store_6'] = bitarr[6]
    testd.loc[mask3, 'Store_7'] = bitarr[7]
    testd.loc[mask3, 'Store_8'] = bitarr[8]
    testd.loc[mask3, 'Store_9'] = bitarr[9]
    testd.loc[mask3, 'Store_10'] = bitarr[10]

# DayOfWeek
traind.insert(len(traind.columns), 'Monday', traind['DayOfWeek'] == 1)
traind.insert(len(traind.columns), 'Tuesday', traind['DayOfWeek'] == 2)
traind.insert(len(traind.columns), 'Wednesday', traind['DayOfWeek'] == 3)
traind.insert(len(traind.columns), 'Thursday', traind['DayOfWeek'] == 4)
traind.insert(len(traind.columns), 'Friday', traind['DayOfWeek'] == 5)
traind.insert(len(traind.columns), 'Saturday', traind['DayOfWeek'] == 6)
traind.insert(len(traind.columns), 'Sunday', traind['DayOfWeek'] == 7)

vald.insert(len(vald.columns), 'Monday', vald['DayOfWeek'] == 1)
vald.insert(len(vald.columns), 'Tuesday', vald['DayOfWeek'] == 2)
vald.insert(len(vald.columns), 'Wednesday', vald['DayOfWeek'] == 3)
vald.insert(len(vald.columns), 'Thursday', vald['DayOfWeek'] == 4)
vald.insert(len(vald.columns), 'Friday', vald['DayOfWeek'] == 5)
vald.insert(len(vald.columns), 'Saturday', vald['DayOfWeek'] == 6)
vald.insert(len(vald.columns), 'Sunday', vald['DayOfWeek'] == 7)

testd.insert(len(testd.columns), 'Monday', testd['DayOfWeek'] == 1)
testd.insert(len(testd.columns), 'Tuesday', testd['DayOfWeek'] == 2)
testd.insert(len(testd.columns), 'Wednesday', testd['DayOfWeek'] == 3)
testd.insert(len(testd.columns), 'Thursday', testd['DayOfWeek'] == 4)
testd.insert(len(testd.columns), 'Friday', testd['DayOfWeek'] == 5)
testd.insert(len(testd.columns), 'Saturday', testd['DayOfWeek'] == 6)
testd.insert(len(testd.columns), 'Sunday', testd['DayOfWeek'] == 7)

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  9, 17)
date_normer = float((date_max - date_min).days)

traind.insert(len(traind.columns), 'Date_Norm', (traind['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
vald.insert(len(vald.columns), 'Date_Norm', (vald['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
testd.insert(len(testd.columns), 'Date_Norm', (testd['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)

# Sales
std = traind['Sales'].std()
mean = traind['Sales'].mean()
traind.loc[:, 'Sales'] = (traind['Sales'] - mean) / std
vald.loc[:, 'Sales'] = (vald['Sales'] - mean) / std

# Customers
del traind['Customers']
del vald['Customers']

# Open

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
traind.insert(len(traind.columns), 'NoCompetition', False)
vald.insert(len(vald.columns), 'NoCompetition', False)
testd.insert(len(testd.columns), 'NoCompetition', False)

traind.loc[traind['HasCompetition'] == 0, 'NoCompetition'] = True
vald.loc[vald['HasCompetition'] == 0, 'NoCompetition'] = True
testd.loc[testd['HasCompetition'] == 0, 'NoCompetition'] = True

# CompetitionDistance
max_comp_dist = traind['CompetitionDistance'].max()

traind.loc[traind['HasCompetition'] == 0, 'CompetitionDistance'] = 0.
vald.loc[vald['HasCompetition'] == 0, 'CompetitionDistance'] = 0.
testd.loc[testd['HasCompetition'] == 0, 'CompetitionDistance'] = 0.

traind.loc[:, 'CompetitionDistance'] /= max_comp_dist
vald.loc[:, 'CompetitionDistance'] /= max_comp_dist
testd.loc[:, 'CompetitionDistance'] /= max_comp_dist

# IsDoingPromo2

print 'Preprocessing 50%% done'

idtrain = traind['Store'].as_matrix().astype(int, copy=False)
xtrain = traind.drop(['Store', 'Date', 'DayOfWeek', 'Open', 'Sales'], axis=1).as_matrix().astype(float, copy=False)
ytrain = traind['Sales'].as_matrix().astype(float, copy=False)
cont_train = np.ones(len(xtrain))

cont_train[0] = 0
for i in range(1, len(idtrain)):
    if idtrain[i] != idtrain[i - 1]:
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
        'mean': mean,
        'std': std
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

print 'Preprocessing 75%% done'

idval = vald['Store'].as_matrix().astype(int, copy=False)
xval = vald.drop(['Store', 'Date', 'DayOfWeek', 'Open', 'Sales'], axis=1).as_matrix().astype(float, copy=False)
yval = vald['Sales'].as_matrix().astype(float, copy=False)
cont_val = np.ones(len(xval))

cont_val[0] = 0
for i in range(1, len(idval)):
    if idval[i] != idval[i - 1]:
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

# When stores are closed
zeros = testd.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

testd_filtered = testd.loc[days_open]
submid = testd_filtered['Id'].as_matrix().astype(int, copy=False)
idtest = testd_filtered['Store'].as_matrix().astype(int, copy=False)
xtest = testd_filtered.drop(['Id', 'Store', 'Date', 'DayOfWeek', 'Open'], axis=1).as_matrix().astype(float, copy=False)
cont_test = np.ones(len(xtest))

cont_test[0] = 0
for i in range(1, len(idtest)):
    if idtest[i] != idtest[i - 1]:
        cont_test[i] = 0

assert len(submid) == len(idtest) == len(xtest) == len(cont_test)
prev_len = len(xtest)
num_rem = batch_size - prev_len % batch_size
idtest = np.insert(idtest, prev_len, idtest[:num_rem], 0)
xtest = np.insert(xtest, prev_len, xtest[:num_rem], 0)

print 'Preprocessing test dataset 50%% done'

# Don't extend submid
cont_test = np.insert(cont_test, prev_len, cont_test[:num_rem], 0)

with open('testd.pickle', 'wb') as f:
    pickle.dump({
        'idtest': idtest,
        'xtest': xtest,
        'submid': submid,
        'cont_test': cont_test,
        'zeros': zeros,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

print 'Preprocessing test dataset done!'

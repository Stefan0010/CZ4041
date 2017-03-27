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
batch_size = 500

# Load train & test data
data_dir = '../data'

train_data = util.load_train_data(data_dir)
test_data = util.load_test_data(data_dir)

# Sort
train_data.sort_values(['Date', 'Store'], ascending=True, inplace=True)
test_data.sort_values(['Date', 'Store'], ascending=True, inplace=True)

# Drop all when stores are closed
train_data = train_data[train_data['Open'] == 1]
days_open = test_data['Open'] == 1

# Store
store_ids = test_data['Store'].unique().astype(int, copy=False)
stores = {}

# DayOfWeek
train_data.loc[:, 'DayOfWeek'] = (train_data['DayOfWeek'] - 1.) / 6.
test_data.loc[:, 'DayOfWeek'] = (test_data['DayOfWeek'] - 1.) / 6.

# Date
date_min = datetime(2013,  1,  1)
date_max = datetime(2015,  9, 18)
date_normer = float((date_max - date_min).days)

train_data.insert(len(train_data.columns), 'Date_Norm', (train_data['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)
test_data.insert(len(test_data.columns), 'Date_Norm', (test_data['Date'] - date_min) / np.timedelta64(1, 'D') / date_normer)

# Sales
for store_id in store_ids:
    mask = train_data['Store'] == store_id
    mean = train_data.loc[mask, 'Sales'].mean()
    std = train_data.loc[mask, 'Sales'].std()
    train_data.loc[mask, 'Sales'] = (train_data.loc[mask, 'Sales'] - mean) / std
    stores[store_id] = {
        'mean': mean,
        'std': std
    }

# Customers
del train_data['Customers']

# Open

# Promo

# StateHoliday
del train_data['StateHoliday']
del test_data['StateHoliday']

# SchoolHoliday

# StateHoliday_0

# StateHoliday_a

# StateHoliday_b

# StateHoliday_c

# Weekends

# Weekdays

# StoreType_a
del train_data['StoreType_a']
del test_data['StoreType_a']

# StoreType_b
del train_data['StoreType_b']
del test_data['StoreType_b']

# StoreType_c
del train_data['StoreType_c']
del test_data['StoreType_c']

# StoreType_d
del train_data['StoreType_d']
del test_data['StoreType_d']

# Assortment_a
del train_data['Assortment_a']
del test_data['Assortment_a']

# Assortment_b
del train_data['Assortment_b']
del test_data['Assortment_b']

# Assortment_c
del train_data['Assortment_c']
del test_data['Assortment_c']

# HasCompetition

# CompetitionDistance
max_compdist = train_data['CompetitionDistance'].max()
train_data.loc[:, 'CompetitionDistance'] = train_data['CompetitionDistance'] / max_compdist
test_data.loc[:, 'CompetitionDistance'] = test_data['CompetitionDistance'] / max_compdist

# IsDoingPromo2

for store_id in store_ids:
    # Drops store id
    dfv = train_data.loc[train_data['Store'] == store_id]
    x = dfv.drop(['Store', 'Date', 'Sales', 'Open'], axis=1).as_matrix().astype(float, copy=False)
    y = dfv['Sales'].as_matrix().astype(float, copy=False)
    del dfv

    # Tweaks for batched learning
    assert len(x) == len(y)
    prev_len = len(x)
    num_rem = batch_size - prev_len % batch_size
    x = np.insert(x, prev_len, x[:num_rem], 0)
    y = np.insert(y, prev_len, y[:num_rem], 0)
    cont = np.ones(len(x))
    cont[0] = 0
    cont[prev_len] = 0

    stores[store_id]['x'] = x
    stores[store_id]['y'] = y
    stores[store_id]['cont'] = cont

    # Process its respective test data
    tempdf = test_data.loc[days_open & (test_data['Store'] == store_id)]
    xtest = tempdf.drop(['Id', 'Store', 'Date', 'Open'], axis=1).as_matrix().astype(float, copy=False)
    submid = tempdf['Id'].as_matrix().astype(int, copy=False)
    del tempdf

    # Tweak for batched learning
    prev_len = len(xtest)
    num_rem = batch_size - prev_len % batch_size

    # Holy fuck
    if prev_len < num_rem:
        cont_test = np.ones(batch_size)
        cont_test[0] = 0
        for i in range(0, num_rem, prev_len):
            xtest = np.insert(xtest, len(xtest), xtest[:min(prev_len, num_rem - i)], 0)
            cont_test[prev_len + i] = 0

    else:
        xtest = np.insert(xtest, len(xtest), xtest[:num_rem], 0)
        cont_test = np.ones(len(xtest))
        cont_test[0] = 0
        cont_test[prev_len] = 0

    stores[store_id]['xtest'] = xtest
    stores[store_id]['submid'] = submid
    stores[store_id]['cont_test'] = cont_test

zeroes = test_data.loc[~days_open, 'Id'].as_matrix().astype(int, copy=False)

# Save that precious RAM,
# do not delete test_data
del train_data

with open('dsetv3.pickle', 'wb') as f:
    pickle.dump({
        'stores': stores,
        'zeroes': zeroes
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

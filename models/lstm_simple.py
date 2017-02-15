import os
import sys
import caffe
import numpy as np
import pandas as pd
from src import util

solver = caffe.SGDSolver('model/lstm_simple_solver.prototxt')

train_data, val_data = util.load_splitted_data()

# Drop all when stores are closed
train_data = train_data.loc[train_data['Open'] == 1]

del train_data['Open']
del train_data['Date']
del train_data['StateHoliday']
train_data.insert(len(train_data.columns), 'DayOfWeek_1', train_data['DayOfWeek'] == 1)
train_data.insert(len(train_data.columns), 'DayOfWeek_2', train_data['DayOfWeek'] == 2)
train_data.insert(len(train_data.columns), 'DayOfWeek_3', train_data['DayOfWeek'] == 3)
train_data.insert(len(train_data.columns), 'DayOfWeek_4', train_data['DayOfWeek'] == 4)
train_data.insert(len(train_data.columns), 'DayOfWeek_5', train_data['DayOfWeek'] == 5)
train_data.insert(len(train_data.columns), 'DayOfWeek_6', train_data['DayOfWeek'] == 6)
train_data.insert(len(train_data.columns), 'DayOfWeek_7', train_data['DayOfWeek'] == 7)
del train_data['DayOfWeek']

# This may be a terrible idea but what the hell
train_data.loc[train_data['CompetitionDistance'] == 0] = train_data['CompetitionDistance'].max()

# -1 to 1 normalization
sales_min = train_data['Sales'].min()
sales_max = train_data['Sales'].max()
train_data.loc[:, 'Sales'] = (train_data['Sales'] - sales_min) * 2. / (sales_max - sales_min) - 1.

# -1 to 1 normalization
customer_min = train_data['Customers'].min()
customer_max = train_data['Customers'].max()
train_data.loc[:, 'Customers'] = (train_data['Customers'] - customer_min) * 2. / (customer_max - customer_min) - 1.

# -1 to 1 normalization
distance_max = train_data['CompetitionDistance'].max()
distance_min = train_data['CompetitionDistance'].min()
train_data.loc[:, 'CompetitionDistance'] = (train_data['CompetitionDistance'] - distance_min) * 2. / (distance_max - distance_min) - 1.

store_ids = train_data['Store'].unique()
stores = {}

for store_id in store_ids:
    mat = train_data.loc[train_data['Store'] == store_id].as_matrix()
    mat = mat[:, 1:].astype(float)

    # Col 0, 1, 18 is not boolean
    mat[:, 2:18] = mat[:, 2:18] * 2. - 1.
    mat[:, 19:] = mat[:, 19:] * 2. - 1.

    stores[store_id] = mat

# Save that precious RAM
del train_data

num_epoch = 3
for epoch in range(num_epoch):
    for store_id in store_ids:
        pass

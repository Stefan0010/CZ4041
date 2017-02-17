import os
import sys
import caffe
import numpy as np
import pandas as pd
from src import util
import matplotlib.pyplot as plt

solver = caffe.get_solver('lstm_simple_solver.prototxt')

data_dir = '../data'
train_data, val_data = util.load_splitted_data(data_dir)

# Drop all when stores are closed
train_data = train_data[train_data['Open'] == 1]

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
train_data.loc[train_data['CompetitionDistance'] == 0, 'CompetitionDistance'] = train_data['CompetitionDistance'].max()

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
store_ids = store_ids.astype(int, copy=False)
stores = {}

for store_id in store_ids:
    mat = train_data.loc[train_data['Store'] == store_id].as_matrix()
    mat = mat[:, 1:].astype(float, copy=False)

    # Col 0, 1, 18 is not boolean
    mat[:, 2:18] = mat[:, 2:18] * 2. - 1.
    mat[:, 19:] = mat[:, 19:] * 2. - 1.

    stores[store_id] = (mat[:, 1:], mat[:, 0])

# Save that precious RAM
del train_data

net = solver.net
test_net = solver.test_nets[0]

# Just a peek
s = store_ids[0]
X, y = stores[s]
ts = X.shape[0]
preds = np.zeros(ts)
plt.figure(1)
plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
for t in range(ts):
    test_net.blobs['data'].data[0] = X[t]
    test_net.blobs['cont'].data[0] = t > 0
    preds[t] = test_net.forward()['ip1'][0]
plt.xlim([0, ts])
y_min=min(preds.min(), y.min())
y_max=max(preds.max(), y.max())
plt.ylim([y_min, y_max])
plt.plot(np.arange(ts), y, '-b')
plt.plot(np.arange(ts), preds, '-r')
plt.show()

raw_input("Start?")

# Start training
num_epoch = 1
for epoch in range(num_epoch):
    for store_id in store_ids:
        X, y = stores[store_id]
        ts = X.shape[0]
        for t in range(ts):
            net.blobs['data'].data[0] = X[t]
            net.blobs['cont'].data[0] = t > 0
            net.blobs['target'].data[0] = y[t]
            solver.step(1)

# Peek again after prediction
s = store_ids[0]
X, y = stores[s]
ts = X.shape[0]
preds = np.zeros(ts)
plt.figure(1)
plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
for t in range(ts):
    test_net.blobs['data'].data[0] = X[t]
    test_net.blobs['cont'].data[0] = t > 0
    preds[t] = test_net.forward()['ip1'][0]

plt.xlim([0, ts])
y_min=min(preds.min(), y.min())
y_max=max(preds.max(), y.max())
plt.ylim([y_min, y_max])
plt.plot(np.arange(ts), y, '-b')
plt.plot(np.arange(ts), preds, '-r')
plt.show()

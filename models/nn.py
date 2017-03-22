import os
import sys
import caffe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

solver = caffe.get_solver('nn_3_layer_solver.prototxt')

data_dir = '../data'
train_data, val_data = util.load_splitted_data(data_dir)

# Drop all when stores are closed
train_data = train_data[train_data['Open'] == 1]
train_data.sort_values(['Store', 'Date'], axis=0, inplace=True)

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

# 0 to 1 normalization
sales_min = train_data['Sales'].min()
sales_max = train_data['Sales'].max()
train_data.loc[:, 'Sales'] = (train_data['Sales'] - sales_min) / (sales_max - sales_min)

# 0 to 1 normalization
customer_min = train_data['Customers'].min()
customer_max = train_data['Customers'].max()
train_data.loc[:, 'Customers'] = (train_data['Customers'] - customer_min) / (customer_max - customer_min)

# 0 to 1 normalization
distance_max = train_data['CompetitionDistance'].max()
distance_min = train_data['CompetitionDistance'].min()
train_data.loc[:, 'CompetitionDistance'] = (train_data['CompetitionDistance'] - distance_min) / (distance_max - distance_min)

store_ids = train_data['Store'].unique()
store_ids = store_ids.astype(int, copy=False)

train_mat = train_data.as_matrix()
X = train_mat[:, 2:].astype(float, copy=False)
y = train_mat[:, 1].astype(float, copy=False)

# Save that precious RAM
del train_data
del train_mat

net = solver.net
test_net = solver.test_nets[0]

# Start training
num_epoch = 15
batch_size = 496
num_rows = X.shape[0]
num_iters = num_rows / batch_size * num_epoch
loss = np.zeros(num_iters)
it = 0
for epoch in range(num_epoch):
    for idx in range(0, num_rows, batch_size):
        net.blobs['data'].data[:] = X[idx:idx+batch_size]
        net.blobs['target'].data[:] = y[idx:idx+batch_size, None]
        solver.step(1)

        loss[it] = net.blobs['loss'].data
        it += 1

plt.figure(1)
plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
plt.xlim([0, num_iters])
plt.ylim([loss.min(), loss.max()])
plt.plot(np.arange(num_iters), loss, '-b')
plt.show()

# Peek again after prediction
idx = 0
plt.figure(1)
plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
test_net.blobs['data'].data[:] = X[idx:idx+batch_size]
preds = test_net.forward()['ip2'][:]

plt.xlim([0, batch_size])
y_min=min(preds.min(), y[idx:idx+batch_size].min())
y_max=max(preds.max(), y[idx:idx+batch_size].max())
plt.ylim([y_min, y_max])
plt.plot(np.arange(batch_size), y[idx:idx+batch_size], '-b')
plt.plot(np.arange(batch_size), preds, '-r')
plt.show()

L = ((preds.ravel() - y[idx:idx+batch_size].ravel())**2).sum() * sales_max * sales_max
print L
print np.sqrt(L)

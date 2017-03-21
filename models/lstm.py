import os
import sys
import caffe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

solver = caffe.get_solver('lstm_simple_solver.prototxt')

data_dir = '../data'
train_data = util.load_train_data(data_dir)

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

sales_std = train_data['Sales'].std()
sales_mean = train_data['Sales'].mean()
train_data.loc[:, 'Sales'] = (train_data['Sales'] - sales_mean) / sales_std

customer_std = train_data['Customers'].std()
customer_mean = train_data['Customers'].mean()
train_data.loc[:, 'Customers'] = (train_data['Customers'] - customer_mean) / customer_std

distance_std = train_data['CompetitionDistance'].std()
distance_mean = train_data['CompetitionDistance'].mean()
train_data.loc[:, 'CompetitionDistance'] = (train_data['CompetitionDistance'] - distance_mean) / distance_std

store_ids = train_data['Store'].unique()
store_ids = store_ids.astype(int, copy=False)
stores = {}

for store_id in store_ids:
    mat = train_data.loc[train_data['Store'] == store_id].as_matrix()
    mat = mat[:, 1:].astype(float, copy=False)

    # Col 0, 1, 18 is not boolean
    # mat[:, 2:18] = mat[:, 2:18] * 2. - 1.
    # mat[:, 19:] = mat[:, 19:] * 2. - 1.

    stores[store_id] = (mat[:, 1:], mat[:, 0])

# Save that precious RAM
del train_data

net = solver.net
test_net = solver.test_nets[0]

# Start training
num_epoch = 2
losses = []
for epoch in range(num_epoch):
    for store_id in store_ids:
        X, y = stores[store_id]
        ts = X.shape[0]
        for t in range(ts):
            net.blobs['data'].data[0] = X[t]
            net.blobs['cont'].data[0] = t > 0
            net.blobs['target'].data[0] = y[t]
            solver.step(1)

            losses.append(net.blobs['loss'].data.item(0))

losses = np.array(losses, dtype=float)
plt.clf()
plt.figure(1)
plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
plt.xlim([0, len(losses[::100])])
plt.ylim([losses[::100].min(), losses[::100].max()])
plt.plot(np.arange(len(losses[::100])), losses[::100], '-b')
plt.show()

# Peek again after training
s = store_ids[0]
X, y = stores[s]

ts = X.shape[0]
preds = np.zeros(ts)
plt.clf()
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

y = sales_mean + y * sales_std
preds = sales_mean + preds * sales_std

RMSE = np.sqrt( ((preds.ravel() - y.ravel())**2).sum()/ts )
RMSPE = np.sqrt( (((preds.ravel() - y.ravel())/y.ravel())**2).sum()/ts )
print 'RMSE  : %.9f' % RMSE
print 'RMSPE : %.9f' % RMSPE

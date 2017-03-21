import os
import sys
import caffe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

batch_size = 100

data_dir = '../data'
train_data = util.load_train_data(data_dir)
test_data = util.load_test_data(data_dir)

# Sort
train_data.sort_values('Date', inplace=True)
test_data.sort_values('Date', inplace=True)

# Drop all when stores are closed
train_data = train_data[train_data['Open'] == 1]

# Not sure what can I do with this
del train_data['Customers']

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

# Do the same thing for test_data
del test_data['Date']
del test_data['StateHoliday']
test_data.insert(len(test_data.columns), 'DayOfWeek_1', test_data['DayOfWeek'] == 1)
test_data.insert(len(test_data.columns), 'DayOfWeek_2', test_data['DayOfWeek'] == 2)
test_data.insert(len(test_data.columns), 'DayOfWeek_3', test_data['DayOfWeek'] == 3)
test_data.insert(len(test_data.columns), 'DayOfWeek_4', test_data['DayOfWeek'] == 4)
test_data.insert(len(test_data.columns), 'DayOfWeek_5', test_data['DayOfWeek'] == 5)
test_data.insert(len(test_data.columns), 'DayOfWeek_6', test_data['DayOfWeek'] == 6)
test_data.insert(len(test_data.columns), 'DayOfWeek_7', test_data['DayOfWeek'] == 7)
del test_data['DayOfWeek']

store_ids = test_data['Store'].unique()
store_ids = store_ids.astype(int, copy=False)
stores = {}
tests = {}

# Insignificant
del train_data['HasCompetition']
del train_data['CompetitionDistance']

del test_data['HasCompetition']
del test_data['CompetitionDistance']

# These features are insignificant
# del train_data['Weekdays']
# del train_data['Weekends']
del train_data['StateHoliday_0']
del train_data['StateHoliday_a']
del train_data['StateHoliday_b']
del train_data['StateHoliday_c']

days_open = test_data['Open'] == 1
for store_id in store_ids:
    mat = train_data.loc[train_data['Store'] == store_id].as_matrix()
    mat = mat[:, 1:].astype(float, copy=False)

    sales_std = mat[:, 0].std()
    sales_mean = mat[:, 0].mean()
    mat[:, 0] = (mat[:, 0] - sales_mean) / sales_std

    mat[:, 1:] = mat[:, 1:] * 2. - 1.

    # Tweaks for batched learning
    prev_len = len(mat)
    num_rem = batch_size - prev_len % batch_size
    mat = np.insert(mat, prev_len, mat[:num_rem], 0)
    cont = np.ones(len(mat))
    cont[0] = 0
    cont[prev_len] = 0

    stores[store_id] = {
        'x': mat[:, 1:],
        'y': mat[:, 0],
        'cont': cont,
        'mean': sales_mean,
        'std': sales_std,
        'prev_len': prev_len
    }

    # Load its respective test data
    temp_df= test_data.loc[days_open & (test_data['Store'] == store_id)]

    # Get rid of unimportant columns
    del temp_df['Open']
    del temp_df['Store']

    del temp_df['StateHoliday_0']
    del temp_df['StateHoliday_a']
    del temp_df['StateHoliday_b']
    del temp_df['StateHoliday_c']

    mat2 = temp_df.as_matrix()
    mat2 = mat2.astype(float, copy=False)

    del temp_df

    # Tweak for batched learning
    prev_len = len(mat2)
    num_rem = batch_size - prev_len % batch_size

    # Holy fuck
    if prev_len < num_rem:
        cont2 = np.ones(batch_size)
        cont2[0] = 0
        for i in range(0, num_rem, prev_len):
            mat2 = np.insert(mat2, len(mat2), mat2[:min(prev_len, num_rem - i)], 0)
            cont2[prev_len + i] = 0

    else:
        mat2 = np.insert(mat2, prev_len, mat2[:num_rem], 0)
        cont2 = np.ones(len(mat2))
        cont[0] = 0
        cont[prev_len] = 0

    tests[store_id] = {
        'submission_ids': mat2[:, 0],
        'x': mat2[:, 1:],
        'cont': cont2,
        'mean': sales_mean,
        'std': sales_std,
        'prev_len': prev_len
    }

# Save that precious RAM,
# do not delete test_data
del train_data

def train(store_id, num_epoch=200, solver='lstm_per_store_solver.prototxt'):
    print '+=============================================================================+'
    print '| Traininig store % 4d                                                        |' % store_id
    print '+=============================================================================+'

    solver = caffe.get_solver(solver)

    net = solver.net
    test_net = solver.test_nets[0]
    losses = []

    x = stores[store_id]['x']
    y = stores[store_id]['y']
    cont = stores[store_id]['cont']
    sales_mean = stores[store_id]['mean']
    sales_std = stores[store_id]['std']
    num_ts = len(x)

    for epoch in range(num_epoch):
        for t in range(0, num_ts, batch_size):
            net.blobs['data'].data[:, 0] = x[t:t+batch_size]
            net.blobs['cont'].data[:, 0] = cont[t:t+batch_size]
            net.blobs['target'].data[:, 0] = y[t:t+batch_size]
            solver.step(1)

            losses.append(net.blobs['loss'].data.item(0))

    print '+=============================================================================+'
    print '| Testing store % 4d                                                          |' % store_id
    print '+=============================================================================+'

    # Peek again after training
    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        test_net.blobs['data'].data[:, 0] = x[t:t+batch_size]
        test_net.blobs['cont'].data[:, 0] = cont[t:t+batch_size]
        preds[t:t+batch_size] = test_net.forward()['output'][:, 0]

    # Reconstruct
    y = sales_mean + y * sales_std
    preds = sales_mean + preds * sales_std

    # Avoid zero sales
    mask = y >= 1.
    num_samples = mask.sum() # Number of acceptable samples

    RMSE = np.sqrt( ((preds[mask].ravel() - y[mask].ravel())**2).sum()/num_samples )
    RMSPE = np.sqrt( (((preds[mask].ravel() - y[mask].ravel())/y[mask].ravel())**2).sum()/num_samples )
    print 'RMSE  : %.9f' % RMSE
    print 'RMSPE : %.9f' % RMSPE

    # Plot losses
    plt.figure(1)
    plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
    plt.xlim(0, len(losses))
    plt.ylim(min(losses), max(losses))
    plt.plot(np.arange(len(losses)), losses, '-b')

    # Plot prediction vs. actual
    plt.figure(2)
    plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
    plt.xlim(0, num_ts)
    y_min=min(preds.min(), y.min())
    y_max=max(preds.max(), y.max())
    plt.ylim(y_min, y_max)
    plt.plot(np.arange(num_ts), y, '-b')
    plt.plot(np.arange(num_ts), preds, '-r')
    plt.show()

    return solver

train(store_ids[0], 200, 'lstm_per_store_v2_solver.prototxt')

# for store_id in store_ids:
#     solver = train(store_id, 200, 'lstm_per_store_v2_solver.prototxt')
#     test_net = solver.test_nets[0]

#     submission_ids = tests[store_id]['submission_ids']
#     x = tests[store_id]['x']
#     cont = tests[store_id]['cont']
#     prev_len = tests[store_id]['prev_len']
#     sales_mean = tests[store_id]['mean']
#     sales_std = tests[store_id]['std']

#     num_ts = len(x)
#     preds = np.zeros(num_ts)
#     for t in range(0, num_ts, batch_size):
#         test_net.blobs['data'].data[:, 0] = x[t:t+batch_size]
#         test_net.blobs['cont'].data[:, 0] = cont[t:t+batch_size]
#         preds[t:t+batch_size] = test_net.forward()['output'][:, 0]

#     preds = sales_mean + preds * sales_std
#     answer = np.zeros((prev_len, 2))
#     answer[:, 0] = submission_ids[:prev_len]
#     answer[:, 1] = preds[:prev_len]

#     with open('answer.csv', 'a') as outfile:
#         for i in range(prev_len):
#             outfile.write('%d,%.15f\n' % (submission_ids[i], preds[i]))

# # Finally
# with open('answer.csv', 'a') as outfile:
#     for submission_id in test_data.loc[~days_open, 'Id']:
#         outfile.write('%d,%.15f\n' % (submission_id, 0.))

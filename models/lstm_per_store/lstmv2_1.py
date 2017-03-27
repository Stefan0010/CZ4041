import os
import sys

import caffe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pickle
from scipy.stats import ortho_group

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

# Some constants
batch_size = 500
num_hidden = 32

# Load train & test data
data_dir = '../data'

with open('dsetv3.pickle', 'rb') as f:
    dset = pickle.load(f)
    stores = dset['stores']
    zeroes = dset['zeroes']
    del dset

def train(store_id, num_epoch=100, solver_prototxt='solverv2_1.prototxt'):
    print '+=============================================================================+'
    print '| Traininig store % 4d                                                        |' % store_id
    print '+=============================================================================+'

    solver = caffe.get_solver(solver_prototxt)

    # W init for LSTM
    for i in range(0, 4 * num_hidden, num_hidden):
        solver.net.params['lstm1'][0].data[i:i+num_hidden] = ortho_group.rvs(dim=num_hidden)
        solver.net.params['lstm1'][2].data[i:i+num_hidden] = ortho_group.rvs(dim=num_hidden)

    net = solver.net
    test_net = solver.test_nets[0]
    losses = []

    x = stores[store_id]['x']
    y = stores[store_id]['y']
    cont = stores[store_id]['cont']
    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    num_ts = len(x)

    for epoch in range(num_epoch):
        for t in range(0, num_ts, batch_size):
            net.blobs['data'].data[:] = x[t:t+batch_size]
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
        test_net.blobs['data'].data[:] = x[t:t+batch_size]
        test_net.blobs['cont'].data[:, 0] = cont[t:t+batch_size]
        preds[t:t+batch_size] = test_net.forward()['output'][:, 0]

    # Reconstruct
    y_de = mean + y * std
    preds = mean + preds * std

    # Avoid division by small number
    mask = y_de < 1.
    y_de[mask] = 1.

    RMSE  = np.sqrt( np.sum((y_de-preds)**2)/num_ts )
    RMSPE = np.sqrt( np.sum(((y_de-preds)/y_de)**2)/num_ts )
    print 'RMSE  : %.9f' % RMSE
    print 'RMSPE : %.9f' % RMSPE

    # Plot losses
    plt.subplots()
    plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
    plt.xlim(0, len(losses))
    plt.ylim(0, max(losses))
    plt.plot(np.arange(len(losses)), losses)

    # Plot prediction vs. actual
    plt.subplots()
    plt.subplots_adjust(left=0.025, bottom=0.025, right=1.0, top=1.0, wspace=0., hspace=0.)
    plt.xlim(0, num_ts)
    y_min=min(preds.min(), y_de.min())
    y_max=max(preds.max(), y_de.max())
    plt.ylim(y_min, y_max)
    plt.plot(np.arange(num_ts), y_de, '-b')
    plt.plot(np.arange(num_ts), preds, '-r')
    plt.show()

    return solver

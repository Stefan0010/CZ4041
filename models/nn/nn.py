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
batch_size = 10000

# Load train & test data
data_dir = '../../data'

with open('dset.pickle', 'rb') as f:
    dset = pickle.load(f)
    idtrain = dset['idtrain']
    xtrain = dset['xtrain']
    ytrain = dset['ytrain']
    submid = dset['submid']
    idtest = dset['idtest']
    xtest = dset['xtest']
    closed = dset['closed']
    stores = dset['stores']

    assert len(idtrain) == len(xtrain) == len(ytrain)

    y_de = np.zeros(len(ytrain))
    for i in range(len(idtrain)):
        sid = idtrain[i]
        y_de[i] = stores[sid]['mean'] + stores[sid]['std'] * ytrain[i]

    assert len(idtest) == len(xtest)

    del dset

def train(num_epochs=10):
    solver = caffe.get_solver('nn_solv.prototxt')

    net = solver.net
    test_net = solver.test_nets[0]

    losses = []
    min_loss = float('inf')

    num_ts = len(xtrain)
    for epoch in range(num_epochs):
        for t in range(0, num_ts, batch_size):
            net.blobs['data'].data[:] = xtrain[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            loss = net.blobs['loss'].data.item(0)
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                net.save('nn.caffemodel')

    plt.subplots()
    plt.plot(np.arange(len(losses)), losses)
    plt.ylim(0, max(losses))
    plt.xlim(0, len(losses))
    plt.subplots_adjust(left=0.025, right=0.99, top=0.99, bottom=0.025, wspace=0., hspace=0.)

    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        test_net.blobs['data'].data[:] = xtrain[t:t+batch_size]
        preds[t:t+batch_size] = test_net.forward()['output'][:, 0]

    for i in range(num_ts):
        sid = idtrain[i]
        preds[i] = stores[sid]['mean'] + stores[sid]['std'] * preds[i]

    rmse = np.sqrt( np.sum((preds-y_de)**2)/num_ts )

    mask = y_de >= 1.

    rmspe = np.sqrt( np.sum(  ((preds[mask]-y_de[mask])/y_de[mask])**2  )/mask.sum() )

    print 'RMSE : %.15f' % rmse
    print 'RMSPE: %.15f' % rmspe

    plt.show()

    return caffe.Net('nn.prototxt', 'nn.caffemodel', caffe.TEST)

net = train(100)

def test(net):
    with open('submission.csv', 'w') as f:
        for ansid in closed:
            f.write('%d,%.15f\n' % (ansid, 0.))

    num_ts = len(xtest)
    preds = np.zeros(num_ts)

    for t in range(0, num_ts, batch_size):
        net.blobs['data'].data[:] = xtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]

    for i in range(num_ts):
        sid = idtest[i]
        preds[i] = stores[sid]['mean'] + stores[sid]['std'] * preds[i]

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

# net = caffe.Net('nn.prototxt', 'nn.caffemodel', caffe.TEST)
test(net)

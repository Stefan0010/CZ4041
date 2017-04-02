import os
import sys

import caffe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import pickle
from datetime import datetime

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

# Some constants
batch_size = 10000

# Load train & test data
data_dir = '../../data'

with open('dset.pickle', 'rb') as f:
    dset = pickle.load(f)
    mean = dset['mean']
    std = dset['std']

    idtrain = dset['idtrain']
    xtrain = dset['xtrain']
    ytrain = dset['ytrain']
    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.
    assert len(idtrain) == len(xtrain) == len(ytrain)

    idval = dset['idval']
    xval = dset['xval']
    yval = dset['yval']
    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.
    assert len(idval) == len(xval) == len(yval)

    submid = dset['submid']
    idtest = dset['idtest']
    xtest = dset['xtest']
    assert len(idtest) == len(xtest)

    closed = dset['closed']

    del dset

def train(num_epochs=10, val_every=1):
    solver = caffe.get_solver('nn_solv.prototxt')

    net = solver.net
    test_net = solver.test_nets[0]

    train_iter = 0
    train_losses = []

    val_x = []
    val_losses = []
    min_loss = float('inf')

    num_ts = len(xtrain)
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print 'EPOCH %d' % epoch
        for t in range(0, num_ts, batch_size):
            net.blobs['store_id'].data[:, 0]    = idtrain[t:t+batch_size]
            net.blobs['data'].data[:]           = xtrain[t:t+batch_size]
            net.blobs['target'].data[:, 0]      = ytrain[t:t+batch_size]
            solver.step(1)

            train_losses.append(net.blobs['loss'].data.item(0))
            train_iter += 1

        if epoch % val_every == 0:
            val_x.append(train_iter - 1)

            preds = np.zeros(len(xval))
            for t in range(0, len(xval), batch_size):
                test_net.blobs['store_id'].data[:, 0]   = idval[t:t+batch_size]
                test_net.blobs['data'].data[:]          = xval[t:t+batch_size]
                preds[t:t+batch_size] = test_net.forward()['output'][:, 0]
            preds_de = mean + preds * std

            loss = np.sum((yval - preds)**2)/len(xval)
            val_losses.append(loss)

            rmspe = np.sqrt( np.sum( ((yval_de[yval_mask]-preds_de[yval_mask])/yval_de[yval_mask])**2 )/yval_mask.sum() )
            print 'Val 50k samples loss: %.9f, rmspe: %.9f' % (loss, rmspe)

            if loss < min_loss:
                min_loss = loss
                net.save('nn.caffemodel')

    net = caffe.Net('nn.prototxt', 'nn.caffemodel', caffe.TEST)

    # preds = np.zeros(len(xval))
    # for t in range(0, len(xval), batch_size):
    #     test_net.blobs['data'].data[:] = xval[t:t+batch_size]
    #     preds[t:t+batch_size] = test_net.forward()['output'][:, 0]
    # preds_de = mean + preds * std

    # plt.subplots()
    # plt.plot(np.arange(500), yval_de[:500], '-b')
    # plt.plot(np.arange(500), preds_de[:500], '-r')

    plt.subplots()
    plt.plot(np.arange(len(train_losses)), train_losses, '-b')
    plt.plot(val_x, val_losses, '-r')
    plt.show()

    return net

net = train(10, 1)

def test(net):
    with open('submission.csv', 'w') as f:
        for ansid in closed:
            f.write('%d,%.15f\n' % (ansid, 0.))

    num_ts = len(xtest)
    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        net.blobs['store_id'].data[:, 0]    = idtest[t:t+batch_size]
        net.blobs['data'].data[:]           = xtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]
    preds = mean + preds * std

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

# test(net)

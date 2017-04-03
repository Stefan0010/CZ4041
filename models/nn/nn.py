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
batch_size = 5000

# Load train & test data
data_dir = '../../data'

with open('dset.pickle', 'rb') as f:
    dset = pickle.load(f)
    mean = dset['mean']
    std = dset['std']

    idtrain = dset['idtrain']
    day_train = dset['day_train']
    month_train = dset['month_train']
    year_train = dset['year_train']
    dow_train = dset['dow_train']
    sth_train = dset['sth_train']
    stype_train = dset['stype_train']
    atype_train = dset['atype_train']

    xtrain = dset['xtrain']
    ytrain = dset['ytrain']
    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.
    assert len(idtrain) == len(xtrain) == len(ytrain)

    idval = dset['idval']
    day_val = dset['day_val']
    month_val = dset['month_val']
    year_val = dset['year_val']
    dow_val = dset['dow_val']
    sth_val = dset['sth_val']
    stype_val = dset['stype_val']
    atype_val = dset['atype_val']

    xval = dset['xval']
    yval = dset['yval']
    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.
    assert len(idval) == len(xval) == len(yval)

    submid = dset['submid']
    idtest = dset['idtest']
    day_test = dset['day_test']
    month_test = dset['month_test']
    year_test = dset['year_test']
    dow_test = dset['dow_test']
    sth_test = dset['sth_test']
    stype_test = dset['stype_test']
    atype_test = dset['atype_test']

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
        num_pass = 0
        train_loss = 0.
        for t in range(0, num_ts, batch_size):
            net.blobs['store_id'].data[:, 0] = idtrain[t:t+batch_size]
            net.blobs['day'].data[:, 0] = day_train[t:t+batch_size]
            net.blobs['month'].data[:, 0] = month_train[t:t+batch_size]
            net.blobs['year'].data[:, 0] = year_train[t:t+batch_size]
            net.blobs['dow'].data[:, 0] = dow_train[t:t+batch_size]
            net.blobs['sth'].data[:, 0] = sth_train[t:t+batch_size]
            net.blobs['stype'].data[:, 0] = stype_train[t:t+batch_size]
            net.blobs['atype'].data[:, 0] = atype_train[t:t+batch_size]
            net.blobs['data'].data[:] = xtrain[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            num_pass += 1
            train_loss += net.blobs['loss'].data.item(0)
        train_losses.append(train_loss / num_pass)

        if epoch % val_every == 0:
            val_x.append(epoch - 1)

            preds = np.zeros(len(xval))
            for t in range(0, len(xval), batch_size):
                test_net.blobs['store_id'].data[:, 0] = idval[t:t+batch_size]
                test_net.blobs['day'].data[:, 0] = day_val[t:t+batch_size]
                test_net.blobs['month'].data[:, 0] = month_val[t:t+batch_size]
                test_net.blobs['year'].data[:, 0] = year_val[t:t+batch_size]
                test_net.blobs['dow'].data[:, 0] = dow_val[t:t+batch_size]
                test_net.blobs['sth'].data[:, 0] = sth_val[t:t+batch_size]
                test_net.blobs['stype'].data[:, 0] = stype_val[t:t+batch_size]
                test_net.blobs['atype'].data[:, 0] = atype_val[t:t+batch_size]
                test_net.blobs['data'].data[:] = xval[t:t+batch_size]
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

net = train(25, 1)

def test(net):
    with open('submission.csv', 'w') as f:
        for ansid in closed:
            f.write('%d,%.15f\n' % (ansid, 0.))

    num_ts = len(xtest)
    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        net.blobs['store_id'].data[:, 0] = idtest[t:t+batch_size]
        net.blobs['day'].data[:, 0] = day_test[t:t+batch_size]
        net.blobs['month'].data[:, 0] = month_test[t:t+batch_size]
        net.blobs['year'].data[:, 0] = year_test[t:t+batch_size]
        net.blobs['dow'].data[:, 0] = dow_test[t:t+batch_size]
        net.blobs['sth'].data[:, 0] = sth_test[t:t+batch_size]
        net.blobs['stype'].data[:, 0] = stype_test[t:t+batch_size]
        net.blobs['atype'].data[:, 0] = atype_test[t:t+batch_size]
        net.blobs['data'].data[:] = xtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]
    preds = mean + preds * std

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

# test(net)

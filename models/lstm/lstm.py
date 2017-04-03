import os
import sys

import caffe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pickle
import time

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

batch_size = 5000

with open('dset.pickle', 'rb') as f:
    dset = pickle.load(f)

    mean = dset['mean']
    std = dset['std']

    xtrain = dset['xtrain']
    cont_train = dset['cont_train']
    ytrain = dset['ytrain']

    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.

    idtrain = dset['idtrain']
    daytrain = dset['daytrain']
    monthtrain = dset['monthtrain']
    yeartrain = dset['yeartrain']
    dowtrain = dset['dowtrain']
    sthtrain = dset['sthtrain']
    stypetrain = dset['stypetrain']
    atypetrain = dset['atypetrain']

    xval = dset['xval']
    cont_val = dset['cont_val']
    yval = dset['yval']

    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.

    idval = dset['idval']
    dayval = dset['dayval']
    monthval = dset['monthval']
    yearval = dset['yearval']
    dowval = dset['dowval']
    sthval = dset['sthval']
    stypeval = dset['stypeval']
    atypeval = dset['atypeval']

    submid = dset['submid']
    xtest = dset['xtest']
    cont_test = dset['cont_test']
    idtest = dset['idtest']
    daytest = dset['daytest']
    monthtest = dset['monthtest']
    yeartest = dset['yeartest']
    dowtest = dset['dowtest']
    sthtest = dset['sthtest']
    stypetest = dset['stypetest']
    atypetest = dset['atypetest']
    closed = dset['closed']

    del dset

def train(num_epochs=10, val_every=1):
    solver = caffe.get_solver('solver.prototxt')

    net = solver.net
    test_net = solver.test_nets[0]

    train_losses = []

    val_x = []
    val_losses = []
    min_loss = float('inf')

    num_ts = len(xtrain)
    for epoch in range(1, num_epochs+1):
        print 'EPOCH %d' % epoch
        num_pass = 0
        train_loss = 0.
        for t in range(0, num_ts, batch_size):
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
            net.blobs['store_id'].data[:, 0] = idtrain[t:t+batch_size]
            net.blobs['day'].data[:, 0] = daytrain[t:t+batch_size]
            net.blobs['month'].data[:, 0] = monthtrain[t:t+batch_size]
            net.blobs['year'].data[:, 0] = yeartrain[t:t+batch_size]
            net.blobs['dow'].data[:, 0] = dowtrain[t:t+batch_size]
            net.blobs['sth'].data[:, 0] = sthtrain[t:t+batch_size]
            net.blobs['stype'].data[:, 0] = stypetrain[t:t+batch_size]
            net.blobs['atype'].data[:, 0] = atypetrain[t:t+batch_size]
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            num_pass += 1
            train_loss += net.blobs['loss'].data.item(0)
        train_loss /= num_pass
        train_losses.append(train_loss)
        print 'Train loss: %.9f' % (train_loss)

        if train_loss < min_loss:
            min_loss = train_loss
            net.save('lstm.caffemodel')

    net = caffe.Net('lstm.prototxt', 'lstm.caffemodel', caffe.TEST)

    return net

net = train(65, 1)

def test(net):
    ansed = set()
    with open('submission.csv', 'w') as f:
        for ansid in closed:
            f.write('%d,%.15f\n' % (ansid, 0.))
            ansed.add(ansid)

    num_ts = len(xtest)
    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        net.blobs['cont'].data[:, 0] = cont_test[t:t+batch_size]
        net.blobs['store_id'].data[:, 0] = idtest[t:t+batch_size]
        net.blobs['day'].data[:, 0] = daytest[t:t+batch_size]
        net.blobs['month'].data[:, 0] = monthtest[t:t+batch_size]
        net.blobs['year'].data[:, 0] = yeartest[t:t+batch_size]
        net.blobs['dow'].data[:, 0] = dowtest[t:t+batch_size]
        net.blobs['sth'].data[:, 0] = sthtest[t:t+batch_size]
        net.blobs['stype'].data[:, 0] = stypetest[t:t+batch_size]
        net.blobs['atype'].data[:, 0] = atypetest[t:t+batch_size]
        net.blobs['data'].data[:, 0] = xtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]
    preds = mean + preds * std

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            if submid[i] in ansed:
                continue
            f.write('%d,%.15f\n' % (submid[i], preds[i]))
            ansed.add(submid[i])

test(net)

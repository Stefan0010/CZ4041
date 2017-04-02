import os
import sys

os.environ['GLOG_minloglevel'] = '3'
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

with open('stores.pickle', 'rb') as f:
    stores = pickle.load(f)

with open('traind.pickle', 'rb') as f:
    dset = pickle.load(f)
    idtrain = dset['idtrain']
    xtrain = dset['xtrain']
    ytrain = dset['ytrain']

    mean = dset['mean']
    std = dset['std']

    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.

    cont_train = dset['cont_train']
    del dset

with open('vald.pickle', 'rb') as f:
    dset = pickle.load(f)
    idval = dset['idval']
    xval = dset['xval']
    yval = dset['yval']

    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.

    cont_val = dset['cont_val']
    del dset

with open('testd.pickle', 'rb') as f:
    dset = pickle.load(f)
    idtest = dset['idtest']
    xtest = dset['xtest']
    submid = dset['submid']
    cont_test = dset['cont_test']
    zeros = dset['zeros']
    del dset

def train(num_epoch=10, val_every=1):
    solver = caffe.get_solver('solver2.prototxt')

    net = solver.net
    test_net = solver.test_nets[0]

    ts = len(xtrain)
    ts_val = len(xval)

    train_iter = 0
    train_losses = []
    min_loss = float('inf')
    val_x = []
    val_losses = []

    for epoch in range(1,num_epoch+1):
        print 'EPOCH %d' % epoch
        for t in range(0, ts, batch_size):
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            train_losses.append(net.blobs['loss'].data.item(0))
            train_iter += 1

        if epoch % val_every != 0:
            continue

        print '+------------+'
        print '| Validating |'
        print '+------------+'
        preds = np.zeros(ts_val)
        for t in range(0, ts_val, batch_size):
            test_net.blobs['data'].data[:, 0] = xval[t:t+batch_size]
            test_net.blobs['cont'].data[:, 0] = cont_val[t:t+batch_size]
            preds[t:t+batch_size] = test_net.forward()['output'][:, 0]
        loss = np.sum((preds - yval)**2)/ts_val

        val_x.append(train_iter - 1)
        val_losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            net.save('lstm2.caffemodel')

        preds_de = mean + preds * std
        rmspe = np.sqrt( np.sum( ((preds_de[yval_mask]-yval_de[yval_mask])/yval_de[yval_mask])**2 )/yval_mask.sum() )

        print 'loss: %.9f, rmspe: %.9f' % (loss, rmspe)

    plt.subplots()
    plt.plot(np.arange(len(train_losses)), train_losses, '-b')
    plt.plot(val_x, val_losses, '-r')
    plt.show()

    return caffe.Net('lstm2.prototxt', 'lstm2.caffemodel', caffe.TEST)

net = train(10)
print 'Training done!'

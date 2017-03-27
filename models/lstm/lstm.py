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

batch_size = 10000

with open('stores.pickle', 'rb') as f:
    stores = pickle.load(f)

with open('traind.pickle', 'rb') as f:
    dset = pickle.load(f)
    idtrain = dset['idtrain']
    xtrain = dset['xtrain']
    ytrain = dset['ytrain']
    cont_train = dset['cont_train']
    del dset

with open('vald.pickle', 'rb') as f:
    dset = pickle.load(f)
    idval = dset['idval']
    xval = dset['xval']
    yval = dset['yval']
    cont_val = dset['cont_val']
    del dset

def train(num_epoch=10, solver_proto='solver2.prototxt'):
    solver = caffe.get_solver(solver_proto)

    net = solver.net
    test_net = solver.test_nets[0]

    ts = len(xtrain)
    ts_val = len(xval)
    losses = []
    for epoch in range(num_epoch):
        print 'EPOCH %d' % (epoch + 1)
        for t in range(0, ts, batch_size):
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            losses.append(net.blobs['loss'].data.item(0))

        print '+------------+'
        print '| Validating |'
        print '+------------+'
        preds = np.zeros(ts_val)
        for t in range(0, ts_val, batch_size):
            test_net.blobs['data'].data[:, 0] = xval[t:t+batch_size]
            test_net.blobs['cont'].data[:, 0] = cont_val[t:t+batch_size]

            preds[t:t+batch_size] = test_net.forward()['output'][:, 0]

        y_de = yval.copy()
        for t in range(ts_val):
            sid = idval[t]
            mean = stores[sid]['mean']
            std = stores[sid]['std']
            preds[t] = mean + preds[t] * std
            y_de[t] = mean + y_de[t] * std
        mask = y_de >= 1.

        RMSE = np.sqrt(  ((preds - y_de)**2).sum()/ts_val  )
        RMSPE = np.sqrt(  (((preds[mask]- y_de[mask])/y_de[mask])**2).sum()/mask.sum()  )
        print 'RMSE : %.9f' % RMSE
        print 'RMSPE: %.9f' % RMSPE

    plt.subplots()
    plt.subplots_adjust(left=0.025, bottom=0.025, right=.99, top=.99, wspace=0., hspace=0.)
    plt.xlim(0, 750)
    plt.plot(np.arange(750), y_de[-750:])
    plt.plot(np.arange(750), preds[-750:])

    plt.subplots()
    plt.subplots_adjust(left=0.025, bottom=0.025, right=.99, top=.99, wspace=0., hspace=0.)
    plt.xlim(0, len(losses))
    plt.ylim(0, max(losses))
    plt.plot(np.arange(len(losses)), losses)
    plt.show()

    return solver

solver = train(5)
print 'Training done!'

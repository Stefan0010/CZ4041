import os
import sys

import caffe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pickle
from scipy.stats import ortho_group
import time

sys.path.append('C:\\Users\\Peter\\Documents\\GitHub\\CZ4041')
from src import util

# Some constants
batch_size = 500

# Load train & test data
data_dir = '../data'

with open('dsetv3.pickle', 'rb') as f:
    dset = pickle.load(f)
    stores = dset['stores']
    zeroes = dset['zeroes']
    del dset

def train(store_id, num_epoch=300, val_every=6):
    net_fname = os.path.join('temp', '_%d.caffemodel' % store_id)

    print '+=============================================================================+'
    print '| Traininig store % 4d                                                        |' % store_id
    print '+=============================================================================+'

    start_time = time.time()

    solver = caffe.get_solver('solverv2.prototxt')
    net = solver.net
    test_net = solver.test_nets[0]
    losses = []

    xtrain = stores[store_id]['xtrain']
    ytrain = stores[store_id]['ytrain']
    cont_train = stores[store_id]['cont_train']
    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.0
    num_ts = len(xtrain)

    min_loss = float('inf')
    for epoch in range(1,num_epoch+1):
        for t in range(0, num_ts, batch_size):
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            losses.append(net.blobs['loss'].data.item(0))

        if epoch % val_every == 0:
            preds = np.zeros(num_ts)
            for t in range(0, num_ts, batch_size):
                test_net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
                test_net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
                preds[t:t+batch_size] = test_net.forward()['output'][:, 0]
            loss = np.sum( (preds-ytrain)**2 )/num_ts

            preds_de = mean + preds * std
            rmspe = np.sqrt( np.sum(((preds_de[ytrain_mask]-ytrain_de[ytrain_mask])/ytrain_de[ytrain_mask])**2)/ytrain_mask.sum() )

            print 'Val 100%% loss: %.9f, rmspe: %.9f' % (loss, rmspe)
            if loss < min_loss:
                min_loss = loss
                solver.net.save(net_fname)

    print '+=============================================================================+'
    # print '| Testing store % 4d, time taken: %.9f' % (store_id, time.time() - start_time)
    print '| Time taken: %.9f' % (time.time() - start_time)
    print '+=============================================================================+'

    plt.subplots()
    plt.plot(np.arange(num_ts), ytrain_de, '-b')
    plt.plot(np.arange(num_ts), preds_de, '-r')
    plt.subplots()
    plt.plot(np.arange(len(losses)), losses)
    plt.show()

    return caffe.Net('lstmv2.prototxt', net_fname, caffe.TEST)

def predict(store_id, net):
    xtest = stores[store_id]['xtest']
    cont_test = stores[store_id]['cont_test']
    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    submid = stores[store_id]['submid']
    num_ts = len(xtest)

    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        net.blobs['data'].data[:, 0] = xtest[t:t+batch_size]
        net.blobs['cont'].data[:, 0] = cont_test[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]

    preds = mean + preds * std

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

with open('submission.csv', 'w') as f:
    for store_id in zeroes:
        f.write('%d,%.15f\n' % (store_id, 0.))

for store_id in stores:
    net_fname = os.path.join('temp', '_%d.caffemodel' % store_id)
    net = train(store_id, 500)
    predict(store_id, net)

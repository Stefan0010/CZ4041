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
batch_size = 50

# Load train & test data
data_dir = '../data'

with open('dsetv3.pickle', 'rb') as f:
    dset = pickle.load(f)
    stores = dset['stores']
    zeroes = dset['zeroes']
    del dset

def train(store_id, num_epoch=200, val_every=5):
    net_fname = os.path.join('temp', '_%d.caffemodel' % store_id)

    print '+=============================================================================+'
    print '| Traininig store % 4d                                                        |' % store_id
    print '+=============================================================================+'

    start_time = time.time()

    solver = caffe.get_solver('solverv2.prototxt')
    net = solver.net
    test_net = solver.test_nets[0]

    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    cont_train = stores[store_id]['cont_train']
    xtrain = stores[store_id]['xtrain']
    ytrain = stores[store_id]['ytrain']
    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.0
    dowtrain = stores[store_id]['dowtrain']
    daytrain = stores[store_id]['daytrain']
    monthtrain = stores[store_id]['monthtrain']
    yeartrain = stores[store_id]['yeartrain']
    sthtrain = stores[store_id]['sthtrain']
    train_iter = 0
    # train_losses = []

    cont_val = stores[store_id]['cont_val']
    xval = stores[store_id]['xval']
    yval = stores[store_id]['yval']
    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.0
    dowval = stores[store_id]['dowval']
    dayval = stores[store_id]['dayval']
    monthval = stores[store_id]['monthval']
    yearval = stores[store_id]['yearval']
    sthval = stores[store_id]['sthval']
    # val_x = []
    # val_losses = []

    num_ts = len(xtrain)

    min_loss = float('inf')
    for epoch in range(1,num_epoch+1):
        for t in range(0, num_ts, batch_size):
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size]
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size]
            net.blobs['day'].data[:, 0] = daytrain[t:t+batch_size]
            net.blobs['month'].data[:, 0] = monthtrain[t:t+batch_size]
            net.blobs['year'].data[:, 0] = yeartrain[t:t+batch_size]
            net.blobs['dow'].data[:, 0] = dowtrain[t:t+batch_size]
            net.blobs['sth'].data[:, 0] = sthtrain[t:t+batch_size]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size]
            solver.step(1)

            # train_losses.append(net.blobs['loss'].data.item(0))
            train_iter += 1

        if epoch % val_every == 0:
            preds = np.zeros(len(xval))
            for t in range(0, len(xval), batch_size):
                test_net.blobs['cont'].data[:, 0] = cont_val[t:t+batch_size]
                test_net.blobs['data'].data[:, 0] = xval[t:t+batch_size]
                test_net.blobs['day'].data[:, 0] = dayval[t:t+batch_size]
                test_net.blobs['month'].data[:, 0] = monthval[t:t+batch_size]
                test_net.blobs['year'].data[:, 0] = yearval[t:t+batch_size]
                test_net.blobs['dow'].data[:, 0] = dowval[t:t+batch_size]
                test_net.blobs['sth'].data[:, 0] = sthval[t:t+batch_size]
                preds[t:t+batch_size] = test_net.forward()['output'][:, 0]
            loss = np.sum( (preds-yval)**2 )/len(xval)

            preds_de = mean + preds * std
            rmspe = np.sqrt( np.sum(((preds_de[yval_mask]-yval_de[yval_mask])/yval_de[yval_mask])**2)/yval_mask.sum() )

            # val_x.append(train_iter - 1)
            # val_losses.append(loss)

            print 'Val 50 samples, loss: %.9f, rmspe: %.9f' % (loss, rmspe)
            if loss < min_loss:
                min_loss = loss
                solver.net.save(net_fname)

    print '+=============================================================================+'
    # print '| Testing store % 4d, time taken: %.9f' % (store_id, time.time() - start_time)
    print '| Time taken: %.9f' % (time.time() - start_time)
    print '+=============================================================================+'

    # plt.subplots()
    # plt.plot(np.arange(len(train_losses)), train_losses, '-b')
    # plt.plot(val_x, val_losses, '-r')
    # plt.show()

    return caffe.Net('lstmv2.prototxt', net_fname, caffe.TEST)

ansed = set()

def predict(store_id, net):
    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    cont_test = stores[store_id]['cont_test']
    xtest = stores[store_id]['xtest']
    daytest = stores[store_id]['daytest']
    monthtest = stores[store_id]['monthtest']
    yeartest = stores[store_id]['yeartest']
    dowtest = stores[store_id]['dowtest']
    sthtest = stores[store_id]['sthtest']
    submid = stores[store_id]['submid']
    num_ts = len(xtest)

    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size):
        net.blobs['cont'].data[:, 0] = cont_test[t:t+batch_size]
        net.blobs['data'].data[:, 0] = xtest[t:t+batch_size]
        net.blobs['day'].data[:, 0] = daytest[t:t+batch_size]
        net.blobs['month'].data[:, 0] = monthtest[t:t+batch_size]
        net.blobs['year'].data[:, 0] = yeartest[t:t+batch_size]
        net.blobs['dow'].data[:, 0] = dowtest[t:t+batch_size]
        net.blobs['sth'].data[:, 0] = sthtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]
    preds = mean + preds * std

    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            if submid[i] in ansed:
                continue
            f.write('%d,%.15f\n' % (submid[i], preds[i]))
            ansed.add(submid[i])

with open('submission.csv', 'w') as f:
    for submid in zeroes:
        f.write('%d,%.15f\n' % (submid, 0.))
        ansed.add(submid)

for store_id in stores:
    net_fname = os.path.join('temp', '_%d.caffemodel' % store_id)
    net = train(store_id, 200, 4)
    predict(store_id, net)

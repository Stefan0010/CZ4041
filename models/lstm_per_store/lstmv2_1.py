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
batch_size = {
    'train': 500,
    'val': 50,
    'test': 50,
}

# Load train & test data
data_dir = '../data'

with open('dsetv4.pickle', 'rb') as f:
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

    solver = caffe.get_solver('solverv2_1.prototxt')
    net = solver.net
    test_net = solver.test_nets[0]

    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    xtrain = stores[store_id]['xtrain']
    ytrain = stores[store_id]['ytrain']
    ytrain_de = mean + ytrain * std
    ytrain_mask = ytrain_de >= 1.0
    cont_train = stores[store_id]['cont_train']
    train_iter = 0
    # train_losses = []

    xval = stores[store_id]['xval']
    yval = stores[store_id]['yval']
    cont_val = stores[store_id]['cont_val']
    yval_de = mean + yval * std
    yval_mask = yval_de >= 1.0
    # val_x = []
    # val_losses = []

    num_ts = len(xtrain)

    min_loss = float('inf')
    for epoch in range(1,num_epoch+1):
        for t in range(0, num_ts, batch_size['train']):
            net.blobs['data'].data[:, 0] = xtrain[t:t+batch_size['train']]
            net.blobs['cont'].data[:, 0] = cont_train[t:t+batch_size['train']]
            net.blobs['target'].data[:, 0] = ytrain[t:t+batch_size['train']]
            solver.step(1)

            # train_losses.append(net.blobs['loss'].data.item(0))
            train_iter += 1

        if epoch % val_every == 0:
            preds = np.zeros(len(xval))
            for t in range(0, len(xval), batch_size['val']):
                test_net.blobs['data'].data[:, 0] = xval[t:t+batch_size['val']]
                test_net.blobs['cont'].data[:, 0] = cont_val[t:t+batch_size['val']]
                preds[t:t+batch_size['val']] = test_net.forward()['output'][:, 0]
            loss = np.sum( (preds-yval)**2 )/len(xval)

            preds_de = mean + preds * std
            rmspe = np.sqrt( np.sum(((preds_de[yval_mask]-yval_de[yval_mask])/yval_de[yval_mask])**2)/yval_mask.sum() )

            # val_x.append(train_iter - 1)
            # val_losses.append(loss)

            # print 'Val 50 samples, loss: %.9f, rmspe: %.9f' % (loss, rmspe)
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

    return caffe.Net('lstmv2_1.prototxt', net_fname, caffe.TEST)

def predict(store_id, net):
    xtest = stores[store_id]['xtest']
    cont_test = stores[store_id]['cont_test']
    mean = stores[store_id]['mean']
    std = stores[store_id]['std']
    submid = stores[store_id]['submid']
    num_ts = len(xtest)

    preds = np.zeros(num_ts)
    for t in range(0, num_ts, batch_size['test']):
        net.blobs['data'].data[:, 0] = xtest[t:t+batch_size['test']]
        net.blobs['cont'].data[:, 0] = cont_test[t:t+batch_size['test']]
        preds[t:t+batch_size['test']] = net.forward()['output'][:, 0]

    preds = mean + preds * std

    with open('submission2.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

with open('submission2.csv', 'w') as f:
    for store_id in zeroes:
        f.write('%d,%.15f\n' % (store_id, 0.))

for store_id in stores:
    net_fname = os.path.join('temp', '_%d.caffemodel' % store_id)
    net = train(store_id, 200, 4)
    predict(store_id, net)

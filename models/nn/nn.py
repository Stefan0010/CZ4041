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
batch_size = 100000

# Load train & test data
data_dir = '../../data'

with open('dset.pickle', 'rb') as f:
    dset = pickle.load(f)
    xtrain = dset['xtrain']
    ytrain = dset['ytrain']
    submid = dset['submid']
    xtest = dset['xtest']
    closed = dset['closed']
    mean = dset['mean']
    std = dset['std']
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
    plt.show()

    return caffe.Net('nn.prototxt', 'nn.caffemodel', caffe.TEST)

net = train(25)

def test(net):
    with open('submission.csv', 'w') as f:
        for ansid in closed:
            f.write('%d,%.15f\n' % (ansid, 0.))

    num_ts = len(xtest)
    preds = np.zeros(num_ts)

    for t in range(0, num_ts, batch_size):
        net.blobs['data'].data[:] = xtest[t:t+batch_size]
        preds[t:t+batch_size] = net.forward()['output'][:, 0]

    preds = preds * std + mean
    with open('submission.csv', 'a') as f:
        for i in range(len(submid)):
            f.write('%d,%.15f\n' % (submid[i], preds[i]))

# net = caffe.Net('nn.prototxt', 'nn.caffemodel', caffe.TEST)
test(net)

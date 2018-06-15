#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
import numpy as np
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

batch_size = 64
img_rows, img_cols = 224, 224
# Data loading and preprocessing
X = np.random.random((batch_size, img_rows, img_cols, 3))
Y = np.random.random((batch_size,)).astype('int32')
Y = to_categorical(Y, 1000)

# Convolutional network building
network = input_data(shape=[None, img_rows, img_cols, 3])
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 1000, activation='softmax')
network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
import time
while True:
    start = time.time()
    model.fit(X, Y, n_epoch=50, snapshot_step=99999)
    print("Time:", time.time() - start)

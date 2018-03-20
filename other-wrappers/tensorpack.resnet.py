#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: tensorpack.resnet.py
import tensorflow as tf
import numpy as np
from tensorpack import *

BATCH = 32  # tensorpack's "batch" is per-GPU batch.
NUM_GPU = 1

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l

def block_func(l, ch_out, stride):
    BN = lambda x, name=None: BatchNorm('bn', x)
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=BN)
    return tf.nn.relu(l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=BN))


def group_func(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 3, 224, 224], 'input'),
                tf.placeholder(tf.int32, [None], 'label') ]

    def build_graph(self, image, label):
        image = image / 255.0

        num_blocks = [3, 4, 6, 3]
        with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]])
                      .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU, padding='VALID')
                      .MaxPooling('pool0', 3, strides=2, padding='SAME')
                      .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                      .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                      .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                      .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, activation=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost')

    def optimizer(self):
        return tf.train.GradientDescentOptimizer(1e-3)


def get_data():
    X_train = np.random.random((BATCH, 3, 224, 224)).astype('float32')
    Y_train = np.random.random((BATCH,)).astype('int32')
    def gen():
        while True:
            yield [X_train, Y_train]
    return DataFromGenerator(gen)

if __name__ == '__main__':
    dataset_train = get_data()
    config = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[],
        max_epoch=100,
        steps_per_epoch=50,
    )
    trainer = SyncMultiGPUTrainerReplicated(
        NUM_GPU, mode='hierarchical' if NR_GPU == 8 else 'cpu')
    launch_train_with_config(config, trainer)

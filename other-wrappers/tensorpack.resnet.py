#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: benchmark-tensorpack.py
import tensorflow as tf
from tensorpack import *

BATCH = 32
NUM_GPU = 1


def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32, name=name)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def block_func(l, ch_out, stride):
    BN = lambda x, name: BatchNorm('bn', x)
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=1, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=BN)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=BN)


def group_func(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label') ]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.transpose(image, [0, 3, 1, 2])
        image = image / 255.0

        num_blocks = [3, 4, 6, 3]
        with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]])
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU, padding='VALID')
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                      .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                      .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                      .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost')

        wrong = prediction_incorrect(logits, label)
        tf.reduce_mean(wrong, name='train_error')
        # no weight decay

    def _get_optimizer(self):
        return tf.train.GradientDescentOptimizer(1e-3)


def get_data(train_or_test):
    return FakeData([[BATCH, 224,224, 3], [BATCH]], dtype=['float32', 'int32'], random=False)

if __name__ == '__main__':
    dataset_train = get_data('train')
    config = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[],
        max_epoch=100,
        steps_per_epoch=50,
    )
    trainer = SyncMultiGPUTrainerReplicated(NUM_GPU)  # change to 8 to benchmark multigpu
    launch_train_with_config(config, trainer)

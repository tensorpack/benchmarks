#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tensorpack.alexnet.py
import tensorflow as tf
import numpy as np
from tensorpack import *

BATCH = 64


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([BATCH, 3, 224, 224], tf.float32, 'input'),
                tf.TensorSpec([BATCH], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 255.0

        with argscope(Conv2D, activation=tf.nn.relu, kernel_size=3), \
                argscope([Conv2D, MaxPooling], data_format='channels_first'):
            logits = (LinearWrap(image)
                      .Conv2D('conv1_1', 64, kernel_size=11, strides=4, padding='VALID')
                      .MaxPooling('pool1', 3, 2)
                      .Conv2D('conv1_2', 192, kernel_size=5)
                      .MaxPooling('pool2', 3, 2)

                      .Conv2D('conv3', 384)
                      .Conv2D('conv4', 256)
                      .Conv2D('conv5', 256)
                      .MaxPooling('pool3', 3, 2)

                      .FullyConnected('fc6', 4096, activation=tf.nn.relu)
                      .Dropout('drop0', rate=0.5)
                      .FullyConnected('fc7', 4096, activation=tf.nn.relu)
                      .Dropout('drop1', rate=0.5)
                      .FullyConnected('fc8', 1000, activation=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cost')
        return cost

    def optimizer(self):
        return tf.train.RMSPropOptimizer(1e-3, epsilon=1e-8)


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
        data=StagingInput(QueueInput(dataset_train)),
        callbacks=[],
        max_epoch=100,
        steps_per_epoch=200,
    )
    launch_train_with_config(config, SimpleTrainer())

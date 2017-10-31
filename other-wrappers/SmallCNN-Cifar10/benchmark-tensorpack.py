#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: benchmark-tensorpack.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import tensorflow as tf
from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf

"""
This script benchmarks a tiny CNN training with tensorpack.

It is meant to be an equivalent of the official keras example at:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
with its `data_augmentation` option set to False (so that we only compare the training).

On machine with Tesla M40 + TensorFlow 1.3.0rc1 + cuda 8.0 + cudnn v6,
this script takes 8.5 seconds per epoch.

Keras with TensorFlow backend takes 14 seconds per epoch,
tflearn takes 14 seconds per epoch.

Note that this CNN is too small to fully utilize GPU, and the datasets are
too small to show any copy latency.
Therefore the advantage of tensorpack cannot be fully demonstrated.

I expect a larger performance gap on larger datasets & larger networks.
"""

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label') ]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.transpose(image, [0, 3, 1, 2])
        image = image / 255.0

        with argscope(Conv2D, nl=tf.nn.relu, kernel_shape=3), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 32)
                      .Conv2D('conv1', 32)
                      .MaxPooling('pool0', 2)
                      .Dropout(0.75)    # keras use drop prob, but we use keep prob
                      .Conv2D('conv2', 64)
                      .Conv2D('conv3', 64)
                      .MaxPooling('pool1', 2)
                      .Dropout(0.75)
                      .FullyConnected('fc1', 512, nl=tf.nn.relu)
                      .Dropout(0.5)
                      .FullyConnected('linear', 10, nl=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost')

        wrong = symbf.prediction_incorrect(logits, label)
        tf.reduce_mean(wrong, name='train_error')
        # no weight decay

    def _get_optimizer(self):
        # keras default is 1e-3
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        return tf.train.RMSPropOptimizer(lr, epsilon=1e-8)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    ds = BatchData(ds, 32, remainder=not isTrain)
    return ds

if __name__ == '__main__':
    logger.auto_set_dir('d')
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    config = TrainConfig(
        model=Model(),
        data=QueueInput(dataset_train,
                        queue=tf.FIFOQueue(300, [tf.float32, tf.int32])),
        callbacks=[InferenceRunner(dataset_test, ClassificationError())],
        # keras monitor these two live data during training. do it here (no overhead actually)
        extra_callbacks=[ProgressBar(['cost', 'train_error']), MergeAllSummaries()],
        max_epoch=200,
    )
    launch_train_with_config(config, SimpleTrainer())

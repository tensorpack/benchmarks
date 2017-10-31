#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: benchmark-tensorpack.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import tensorflow as tf
from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf

BATCH = 64

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [BATCH, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [BATCH], 'label') ]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.transpose(image, [0, 3, 1, 2])
        image = image / 255.0

        with argscope(Conv2D, nl=tf.nn.relu, kernel_shape=3), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
            logits = (LinearWrap(image)
                      .Conv2D('conv1_1', 64, kernel_shape=11, stride=4, padding='VALID')
                      .MaxPooling('pool1', 3, 2)
                      .Conv2D('conv1_2', 192, kernel_shape=5)
                      .MaxPooling('pool2', 3, 2)

                      .Conv2D('conv3', 384)
                      .Conv2D('conv4', 256)
                      .Conv2D('conv5', 256)
                      .MaxPooling('pool3', 3, 2)

                      .FullyConnected('fc6', 4096, nl=tf.nn.relu)
                      .Dropout('drop0', 0.5)
                      .FullyConnected('fc7', 4096, nl=tf.nn.relu)
                      .Dropout('drop1', 0.5)
                      .FullyConnected('fc8', out_dim=1000, nl=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost')

        wrong = symbf.prediction_incorrect(logits, label)
        tf.reduce_mean(wrong, name='train_error')
        # no weight decay

    def _get_optimizer(self):
        # keras default is 1e-3
        lr = symbf.get_scalar_var('learning_rate', 1e-3, summary=True)
        return tf.train.RMSPropOptimizer(lr, epsilon=1e-8)


def get_data(train_or_test):
    return FakeData([[BATCH, 224,224, 3], [BATCH]], random=False, dtype=['float32', 'int32'])

if __name__ == '__main__':
    logger.auto_set_dir('d')
    dataset_train = get_data('train')
    config = TrainConfig(
        model=Model(),
        data=QueueInput(dataset_train),
        callbacks=[],
        # keras monitor these two live data during training. do it here (no overhead actually)
        #extra_callbacks=[ProgressBar(['cost', 'train_error'])],
        max_epoch=200,
        steps_per_epoch=200,
    )
    launch_train_with_config(config, SimpleTrainer())

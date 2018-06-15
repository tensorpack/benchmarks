#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tensorpack.cifar10.py
import tensorflow as tf
from tensorpack import *


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = tf.transpose(image, [0, 3, 1, 2])
        image = image / 255.0

        with argscope(Conv2D, activation=tf.nn.relu, kernel_size=3, padding='VALID'), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 32, padding='SAME')
                      .Conv2D('conv1', 32)
                      .MaxPooling('pool0', 2)
                      .Dropout(rate=0.25)
                      .Conv2D('conv2', 64, padding='SAME')
                      .Conv2D('conv3', 64)
                      .MaxPooling('pool1', 2)
                      .Dropout(rate=0.25)
                      .FullyConnected('fc1', 512, activation=tf.nn.relu)
                      .Dropout(rate=0.5)
                      .FullyConnected('linear', 10, activation=tf.identity)())

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong')
        tf.reduce_mean(wrong, name='train_error')
        # no weight decay

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.RMSPropOptimizer(lr, epsilon=1e-8)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    ds = BatchData(ds, 32, remainder=not isTrain)
    return ds


if __name__ == '__main__':
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    config = TrainConfig(
        model=Model(),
        data=QueueInput(dataset_train,
                        queue=tf.FIFOQueue(300, [tf.float32, tf.int32])),
        # callbacks=[InferenceRunner(dataset_test, ClassificationError('wrong'))],   # skip validation
        callbacks=[],
        # keras monitor these two live data during training. do it here (no overhead actually)
        extra_callbacks=[ProgressBar(['cost', 'train_error']), MergeAllSummaries()],
        max_epoch=200,
    )
    launch_train_with_config(config, SimpleTrainer())

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: resnet-tfbench.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from tfbench.convnet import ConvNetBuilder
from tfbench import model_config

INPUT_SHAPE = 224
DEPTH = 50


class Model(ModelDesc):
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


class TFBenchModel(Model):
    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        network = ConvNetBuilder(image, 3, True, data_format=self.data_format)
        model_conf = model_config.get_model_config('resnet50')
        model_conf.set_batch_size(64)
        model_conf.add_inference(network)
        logits = network.affine(1000, activation='linear')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')
        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        #wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        #add_moving_summary(loss, wd_cost)
        wd_cost = 0.0
        self.cost = tf.add_n([loss, wd_cost], name='cost')


class TensorpackModel(Model):
    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
                l = BatchNorm('bnshortcut', l)
                return l
            else:
                return l

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            input = l
            l = Conv2D('conv1', l, ch_out, 1, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=1, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1, nl=tf.identity)
            l = BatchNorm('bn', l)
            ret = l + shortcut(input, ch_in, ch_out * 4, stride)
            return tf.nn.relu(ret)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      #())
            #self.cost = tf.reduce_mean(logits)
            #return
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')


def get_data():
    return FakeData([[64, 224, 224, 3], [64]], 1000, random=False, dtype='float32')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', choices=['tfbench', 'tensorpack'], default='tensorpack')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--send', help='', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    args = parser.parse_args()
    if args.send:
        d = get_data()
        send_dataflow_zmq(d, 'ipc://ipcpipe', format='op')
        sys.exit()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    NR_GPU = get_nr_gpu()

    M = TFBenchModel if args.model == 'tfbench' else TensorpackModel
    config = TrainConfig(
        model=M(data_format=args.data_format),
        dataflow=get_data(),
        callbacks=[ ],
        steps_per_epoch=50,
        max_epoch=10,
        nr_tower=NR_GPU
    )
    gpus = ['/gpu:{}'.format(k) for k in range(NR_GPU)]
    print(gpus)

    config.data = QueueInput(config.dataflow)
    #config.data = DummyConstantInput([[64, 224,224,3],[64]])
    config.dataflow = None
    if NR_GPU == 1:
        SimpleFeedfreeTrainer(config).train()
    else:
        #config.data = StagingInputWrapper(config.data, gpus)
        SyncMultiGPUTrainerReplicated(config, gpu_prefetch=False).train()

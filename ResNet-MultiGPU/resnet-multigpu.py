#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: resnet-multigpu.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.train import HorovodTrainer
from tensorpack.utils.stats import RatioCounter
from tensorpack.utils.gpu import get_nr_gpu
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
        return tf.train.GradientDescentOptimizer(lr)


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
        wd_cost = regularize_cost('.*', tf.contrib.layers.l2_regularizer(1e-4))
        self.cost = tf.add_n([loss, wd_cost], name='cost')


class TensorpackModel(Model):
    """
    Implement the same model with tensorpack layers.
    """
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
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', choices=['tfbench', 'tensorpack'], default='tfbench')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('--fake-location', help='the place to create fake data',
                        type=str, default='gpu', choices=['cpu', 'gpu', 'python'])
    parser.add_argument('--variable-update', help='variable update strategy',
                        type=str, choices=['replicated', 'parameter_server', 'horovod'],
                        required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    NR_GPU = get_nr_gpu()

    M = TFBenchModel if args.model == 'tfbench' else TensorpackModel
    config = TrainConfig(
        model=M(data_format=args.data_format),
        callbacks=[ ],
        steps_per_epoch=100,
        max_epoch=10,
    )

    input_shape = [64, 224, 224, 3]
    label_shape = [64]
    if args.fake_location == 'gpu':
        config.data = DummyConstantInput([input_shape, label_shape])
    elif args.fake_location == 'cpu':
        def fn():
            # these copied from tensorflow/benchmarks
            with tf.device('/cpu:0'):
                images = tf.truncated_normal(
                    input_shape, dtype=tf.float32, stddev=1e-1, name='synthetic_images')
                labels = tf.random_uniform(
                    label_shape, minval=1, maxval=1000, dtype=tf.int32, name='synthetic_labels')
                images = tf.contrib.framework.local_variable(images, name='images')
                labels = tf.contrib.framework.local_variable(labels, name='labels')
                return [images, labels]
        config.data = TensorInput(fn)
    else:
        dataflow = FakeData([input_shape, label_shape], 1000, random=False, dtype='float32')
        # our training is quite fast, so we stage more data than default
        #ds = TFDatasetInput.dataflow_to_dataset(dataflow, [tf.float32, tf.int32])
        #ds = ds.prefetch(30)
        #config.data = TFDatasetInput(ds)
        config.data = QueueInput(dataflow)
        config.data = StagingInputWrapper(
            config.data, range(NR_GPU), nr_stage=max(2 * NR_GPU, 5))

    if NR_GPU == 1:
        trainer = SimpleTrainer()
    else:
        trainer = {
            'replicated': lambda: SyncMultiGPUTrainerReplicated(NR_GPU),
            'horovod': lambda: HorovodTrainer(),
            'parameter_server': lambda: SyncMultiGPUTrainerParameterServer(NR_GPU, ps_device='cpu')
        }[args.variable_update]()
        # we already handle gpu_prefetch above manually.
    launch_train_with_config(config, trainer)

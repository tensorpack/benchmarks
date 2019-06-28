#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from contextlib import contextmanager


from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, FullyConnected, BNReLU, layer_register)


@layer_register(log_shape=False, use_scope=False)
def Norm(x, type, gamma_initializer=tf.constant_initializer(1.)):
    """
    A norm layer (which depends on 'type')

    Args:
        type (str): one of "BN" or "GN"
    """
    assert type in ["BN", "GN"]
    if type == "BN":
        return BatchNorm('bn', x, gamma_initializer=gamma_initializer)
    else:
        return GroupNorm('gn', x, gamma_initializer=gamma_initializer)


@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    shortcut = l
    norm_relu = lambda x: tf.nn.relu(Norm(x))
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=norm_relu)
    """
    Sec 5.1:
    We use the ResNet-50 [16] variant from [12], noting that
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=norm_relu)
    """
    Section 5.1:
    For BN layers, the learnable scaling coefficient γ is initialized
    to be 1, except for each residual block's last BN
    where γ is initialized to be 0.
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=lambda x: Norm(x, gamma_initializer=tf.zeros_initializer()))
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=lambda x: Norm(x))
    return tf.nn.relu(ret, name='block_output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


@contextmanager
def weight_standardization_context(enable):
    if enable:
        def weight_standardization(v):
            if (not v.name.endswith('/W:0')) or v.shape.ndims != 4:
                return v
            mean, var = tf.nn.moments(v, [0, 1, 2], keep_dims=True)
            v = (v - mean) / (tf.sqrt(var) + 1e-5)
            return v

        with remap_variables(weight_standardization):
            yield

    else:
        yield


def resnet_backbone(image, num_blocks, group_func, block_func):
    """
    Sec 5.1: We adopt the initialization of [15] for all convolutional layers.
    TensorFlow does not have the true "MSRA init". We use variance_scaling as an approximation.
    """
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    """
    Sec 5.1:
    The 1000-way fully-connected layer is initialized by
    drawing weights from a zero-mean Gaussian with standard
    deviation of 0.01.
    """
    return logits

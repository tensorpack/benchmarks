#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, FullyConnected)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    """
    Sec 5.1:
    We use the ResNet-50 [16] variant from [12], noting that
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    """
    Section 5.1:
    For BN layers, the learnable scaling coefficient γ is initialized
    to be 1, except for each residual block's last BN
    where γ is initialized to be 0.
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(ret)


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
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

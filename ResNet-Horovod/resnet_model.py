#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def get_bn(zero_init=False):
    """
    Sec 5.1: For BN layers, the learnable scaling coefficient γ is initialized
    to be 1, except for each residual block's last BN
    where γ is initialized to be 0. Setting γ = 0 in the last BN of
    each residual block causes the forward/backward signal initially
    to propagate through the identity shortcut of ResNets,
    which we found to ease optimization at the start of training.
    This initialization improves all models but is particularly
    helpful for large minibatch training as we will show.'
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    """
    Sec 5.1:
    We use the ResNet-50 [16] variant from [12], noting that
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    """
    Section 5.1:
    For BN layers, the learnable scaling coefficient γ is initialized
    to be 1, except for each residual block's last BN
    where γ is initialized to be 0.
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                  .GlobalAvgPooling('gap')
                  .FullyConnected(
                      'linear', 1000, nl=tf.identity,
                      W_init=tf.random_normal_initializer(stddev=0.01))())
    """
    Sec 5.1:
    The 1000-way fully-connected layer is initialized by
    drawing weights from a zero-mean Gaussian with standard
    deviation of 0.01.
    """
    return logits

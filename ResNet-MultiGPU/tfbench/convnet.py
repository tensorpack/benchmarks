#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from collections import defaultdict
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import data_flow_ops
import numpy as np

class ConvNetBuilder(object):
  """Builder of cnn net."""

  def __init__(self,
               input_op,
               input_nchan,
               phase_train,
               data_format='NCHW',
               data_type=tf.float32):
    self.top_layer = input_op
    self.top_size = input_nchan
    self.phase_train = phase_train
    self.data_format = data_format
    self.data_type = data_type
    self.counts = defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = (
        'channels_last' if data_format == 'NHWC' else 'channels_first')

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           batch_norm=None,
           activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = conv_layers.conv2d(
            input_layer,
            num_out_channels, [k_height, k_width],
            strides=[d_height, d_width],
            padding=mode,
            data_format=self.channel_pos,
            use_bias=False)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = conv_layers.conv2d(
              input_layer,
              num_out_channels, [k_height, k_width],
              strides=[d_height, d_width],
              padding='SAME',
              data_format=self.channel_pos,
              use_bias=False)
        else:
          rate = 1  # Unused (for 'a trous' convolutions)
          kernel_size_effective = k_height + (k_width - 1) * (rate - 1)
          pad_total = kernel_size_effective - 1
          pad_beg = pad_total // 2
          pad_end = pad_total - pad_beg
          padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          input_layer = tf.pad(input_layer, padding)
          conv = conv_layers.conv2d(
              input_layer,
              num_out_channels, [k_height, k_width],
              strides=[d_height, d_width],
              padding='VALID',
              data_format=self.channel_pos,
              use_bias=False)
      if batch_norm is None:
        batch_norm = self.use_batch_norm
      if not batch_norm:
        biases = tf.get_variable(
            'biases', [num_out_channels], self.data_type,
            tf.constant_initializer(0.0))
        biased = tf.reshape(
            tf.nn.bias_add(
                conv, biases, data_format=self.data_format),
            conv.get_shape())
      else:
        self.top_layer = conv
        self.top_size = num_out_channels
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        conv1 = tf.nn.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = num_out_channels
      return conv1

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = 'mpool' + str(self.counts['mpool'])
    self.counts['mpool'] += 1
    pool = pooling_layers.max_pooling2d(
        input_layer, [k_height, k_width], [d_height, d_width],
        padding=mode,
        data_format=self.channel_pos,
        name=name)
    self.top_layer = pool
    return pool

  def apool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct an average pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = 'apool' + str(self.counts['apool'])
    self.counts['apool'] += 1
    pool = pooling_layers.average_pooling2d(
        input_layer, [k_height, k_width], [d_height, d_width],
        padding=mode,
        data_format=self.channel_pos,
        name=name)
    self.top_layer = pool
    return pool

  def reshape(self, shape, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    self.top_layer = tf.reshape(input_layer, shape)
    self.top_size = shape[-1]  # HACK This may not always work
    return self.top_layer

  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      kernel = tf.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.data_type,
          tf.random_normal_initializer(stddev=np.sqrt(init_factor /
                                                      (num_channels_in))))
      biases = tf.get_variable('biases', [num_out_channels],
                               self.data_type,
                               tf.constant_initializer(0.0))
      logits = tf.matmul(input_layer, kernel) + biases
      if activation == 'relu':
        affine1 = tf.nn.relu(logits, name=name)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

  def resnet_bottleneck_v1(self,
                           depth,
                           depth_bottleneck,
                           stride,
                           input_layer=None,
                           in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name = 'resnet_v1' + str(self.counts['resnet_v1'])
    self.counts['resnet_v1'] += 1
    with tf.variable_scope(name):
      if depth == in_size:
        if stride == 1:
          shortcut = input_layer
        else:
          assert(False)
          shortcut = self.mpool(
              1,
              1,
              stride,
              stride,
              input_layer=input_layer,
              num_channels_in=in_size)
      else:
        shortcut = self.conv(
            depth,
            1,
            1,
            stride,
            stride,
            activation=None,
            input_layer=input_layer,
            num_channels_in=in_size)
      self.conv(
          depth_bottleneck,
          1,
          1,
          stride,
          stride,
          input_layer=input_layer,
          num_channels_in=in_size)
      self.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET')
      res = self.conv(depth, 1, 1, 1, 1, activation=None)
      output = tf.nn.relu(shortcut + res)
      self.top_layer = output
      self.top_size = depth
      return output

  def inception_module(self, name, cols, input_layer=None, in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name += str(self.counts[name])
    self.counts[name] += 1
    with tf.variable_scope(name):
      col_layers = []
      col_layer_sizes = []
      for c, col in enumerate(cols):
        col_layers.append([])
        col_layer_sizes.append([])
        for l, layer in enumerate(col):
          ltype, args = layer[0], layer[1:]
          kwargs = {
              'input_layer': input_layer,
              'num_channels_in': in_size
          } if l == 0 else {}
          if ltype == 'conv':
            self.conv(*args, **kwargs)
          elif ltype == 'mpool':
            self.mpool(*args, **kwargs)
          elif ltype == 'apool':
            self.apool(*args, **kwargs)
          elif ltype == 'share':  # Share matching layer from previous column
            self.top_layer = col_layers[c - 1][l]
            self.top_size = col_layer_sizes[c - 1][l]
          else:
            raise KeyError('Invalid layer type for inception module: \'%s\'' %
                           ltype)
          col_layers[c].append(self.top_layer)
          col_layer_sizes[c].append(self.top_size)
      catdim = 3 if self.data_format == 'NHWC' else 1
      self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
      self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
      return self.top_layer

  def residual(self, nout, net, scale=1.0):
    inlayer = self.top_layer
    net(self)
    self.conv(nout, 1, 1, activation=None)
    self.top_layer = tf.nn.relu(inlayer + scale * self.top_layer)

  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keep_dims=keep_dims, name=name)
    return self.top_layer

  def dropout(self, keep_prob=0.5, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'dropout' + str(self.counts['dropout'])
    with tf.variable_scope(name):
      if not self.phase_train:
        keep_prob = 1.0
      keep_prob_tensor = tf.constant(keep_prob, dtype=self.data_type)
      dropout = core_layers.dropout(input_layer, keep_prob_tensor)
      self.top_layer = dropout
      return dropout

  def batch_norm(self, input_layer=None, **kwargs):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    with tf.variable_scope(name) as scope:
      bn = tf.contrib.layers.batch_norm(
          input_layer, is_training=self.phase_train,
          fused=True, data_format=self.data_format,
          scope=scope, **kwargs)
    self.top_layer = bn
    return bn

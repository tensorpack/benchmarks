#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: symbolic_imagenet.py

import os
import tensorflow as tf
import numpy as np
from tensorpack.dataflow import dataset
from tensorpack.utils import logger

__all__ = ['get_imglist', 'build_pipeline', 'lighting']


def get_imglist(dir, name):
    """
    Args:
        dir(str): directory which contains name
        name(str): 'train' or 'val'

    Returns:
        [(full filename, label)]
    """
    dir = os.path.join(dir, name)
    meta = dataset.ILSVRCMeta()
    imglist = meta.get_image_list(
        name,
        dataset.ILSVRCMeta.guess_dir_structure(dir))

    def _filter(fname):
        # png
        return 'n02105855_2933.JPEG' in fname

    ret = []
    for fname, label in imglist:
        if _filter(fname):
            logger.info("Image {} was filtered out.".format(fname))
            continue
        fname = os.path.join(dir, fname)
        ret.append((fname, label))
    return ret


def uint8_resize_bicubic(image, shape):
    ret = tf.image.resize_bicubic([image], shape)
    return tf.cast(tf.clip_by_value(ret, 0, 255), tf.uint8)[0]


def resize_shortest_edge(image, image_shape, size):
    shape = tf.cast(image_shape, tf.float32)
    w_greater = tf.greater(image_shape[0], image_shape[1])
    shape = tf.cond(w_greater,
                    lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                    lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

    return uint8_resize_bicubic(image, shape)


def center_crop(image, size):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - size) // 2
    offset_width = (image_width - size) // 2
    image = tf.slice(image, [offset_height, offset_width, 0], [size, size, -1])
    return image


def lighting(image, std, eigval, eigvec):
    v = tf.random_uniform(shape=[3]) * std * eigval
    inc = tf.matmul(eigvec, tf.reshape(v, [3, 1]))
    image = tf.cast(tf.cast(image, tf.float32) + tf.reshape(inc, [3]), image.dtype)
    return image


def training_mapper(filename, label):
    byte = tf.read_file(filename)

    jpeg_opt = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}
    jpeg_shape = tf.image.extract_jpeg_shape(byte)  # hwc
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        jpeg_shape,
        bounding_boxes=tf.zeros(shape=[0, 0, 4]),
        min_object_covered=0,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)

    is_bad = tf.reduce_sum(tf.cast(tf.equal(bbox_size, jpeg_shape), tf.int32)) >= 2

    def good():
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

        image = tf.image.decode_and_crop_jpeg(
            byte, crop_window, channels=3, **jpeg_opt)
        image = uint8_resize_bicubic(image, [224, 224])
        return image

    def bad():
        image = tf.image.decode_jpeg(
            tf.reshape(byte, shape=[]), 3, **jpeg_opt)
        image = resize_shortest_edge(image, jpeg_shape, 224)
        image = center_crop(image, 224)
        return image

    image = tf.cond(is_bad, bad, good)
    # TODO other imgproc
    # image = lighting(image, 0.1,
    #    eigval=np.array([0.2175, 0.0188, 0.0045], dtype='float32') * 255.0,
    #    eigvec=np.array([[-0.5675, 0.7192, 0.4009],
    #                     [-0.5808, -0.0045, -0.8140],
    #                     [-0.5836, -0.6948, 0.4203]], dtype='float32'))
    image = tf.image.random_flip_left_right(image)
    return image, label


def validation_mapper(filename, label):
    byte = tf.read_file(filename)

    jpeg_opt = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}
    image = tf.image.decode_jpeg(
        tf.reshape(byte, shape=[]), 3, **jpeg_opt)
    image = resize_shortest_edge(image, tf.shape(image), 256)
    image = center_crop(image, 224)
    return image, label


def build_pipeline(imglist, training, batch, parallel):
    """
    Args:
        imglist (list): [(full filename, label)]
        training (bool):
        batch (int):
        parallel (int):

    If training, returns an infinite dataset.

    Note that it produces RGB images, not BGR.
    """
    N = len(imglist)
    filenames = tf.constant([k[0] for k in imglist], name='filenames')
    labels = tf.constant([k[1] for k in imglist], dtype=tf.int32, name='labels')

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if training:
        ds = ds.shuffle(N, reshuffle_each_iteration=True).repeat()

    mapper = training_mapper if training else validation_mapper

    ds = ds.apply(
        tf.contrib.data.map_and_batch(
            mapper,
            batch_size=batch,
            num_parallel_batches=parallel))
    ds = ds.prefetch(100)
    return ds

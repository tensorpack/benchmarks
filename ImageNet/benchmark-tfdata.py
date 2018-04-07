#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-tfdata.py

import os
import numpy as np
import tqdm
import argparse
import tensorflow as tf
from tensorpack.dataflow import dataset
from tensorpack.tfutils.common import get_default_sess_config

def get_list(dir, name):
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
            print("{} filtered.".format(fname))
            continue
        fname = os.path.join(dir, fname)
        ret.append((fname, label))
    return ret


def resize_shortest_edge(image, image_shape, size):
    shape = tf.cast(image_shape, tf.float32)
    w_greater = tf.greater(shape[0], shape[1])
    shape = tf.cond(w_greater,
                    lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                    lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

    return tf.image.resize_bicubic([image], shape)[0]


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

def build_pipeline(filenames, labels, batch=128):
    N = len(filenames)
    filenames = tf.constant(filenames, name='filenames')
    labels = tf.constant(labels, name='labels')

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.shuffle(N, reshuffle_each_iteration=True)

    def mapper(filename, label):
        byte = tf.read_file(filename)

        jpeg_opt = {'fancy_upscaling': True, 'dct_method': 'INTEGER_FAST'}
        if False:   # fuse decode & crop
            image = tf.image.decode_jpeg(
                tf.reshape(byte, shape=[]), 3, **jpeg_opt)
            image = tf.image.resize_bilinear([image], [224, 224])[0]
        else:
            jpeg_shape = tf.image.extract_jpeg_shape(byte)  # hwc
            bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                jpeg_shape,
                bounding_boxes=tf.zeros(shape=[0, 1, 4]),
                min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.08, 1.0],
                max_attempts=10,
                use_image_if_no_bounding_boxes=True)
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

            is_bad = tf.reduce_sum(tf.cast(tf.equal(bbox_size, jpeg_shape), tf.int32)) >= 2

            def good():
                image = tf.image.decode_and_crop_jpeg(
                    byte, crop_window, channels=3, **jpeg_opt)
                image = tf.image.resize_bicubic([image], [224, 224])[0]
                return image

            def bad():
                image = tf.image.decode_jpeg(
                    tf.reshape(byte, shape=[]), 3, **jpeg_opt)
                image = resize_shortest_edge(image, jpeg_shape, 224)
                image = center_crop(image, 224)
                return image

            image = tf.cond(is_bad, bad, good)
        # TODO imgproc
        #image = lighting(image, 0.1,
        #    eigval=np.array([0.2175, 0.0188, 0.0045], dtype='float32') * 255.0,
        #    eigvec=np.array([[-0.5675, 0.7192, 0.4009],
        #                     [-0.5808, -0.0045, -0.8140],
        #                     [-0.5836, -0.6948, 0.4203]], dtype='float32'))
        image = tf.image.random_flip_left_right(image)
        return (image, label)

    ds = ds.apply(
        tf.contrib.data.map_and_batch(
            mapper,
            batch_size=128,
            num_parallel_batches=60))
    ds = ds.prefetch(100).repeat()
    return ds


def benchmark_ds(ds, count, warmup=200):
    itr = ds.make_initializable_iterator()
    dp = itr.get_next()
    dpop = tf.group(*dp)
    with tf.Session(config=get_default_sess_config()) as sess:

        sess.run(itr.initializer)
        for _ in tqdm.trange(warmup):
            sess.run(dpop)
        for _ in tqdm.trange(count, smoothing=0.1):
            sess.run(dpop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='directory to imagenet')
    parser.add_argument('--name', choices=['train', 'val'], default='train')
    args = parser.parse_args()

    imglist = get_list(args.data, args.name)
    filenames = [k[0] for k in imglist]
    labels = np.asarray([k[1] for k in imglist], dtype='int32')

    with tf.device('/cpu:0'):
        data = build_pipeline(filenames, labels)
    benchmark_ds(data, 100000)

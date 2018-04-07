#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-tfdata.py

import os
import numpy as np
import tqdm
import argparse
import tensorflow as tf
from tensorpack.tfutils.common import get_default_sess_config

from symbolic_imagenet import get_imglist, build_pipeline


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
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--parallel', type=int, default=60)
    args = parser.parse_args()

    imglist = get_imglist(args.data, args.name)
    print("Number of Images: {}".format(len(imglist)))

    with tf.device('/cpu:0'):
        data = build_pipeline(
            imglist, args.name == 'train',
            args.batch, args.parallel)
        if args.name != 'train':
            data = data.repeat()    # for benchmark
    benchmark_ds(data, 100000)

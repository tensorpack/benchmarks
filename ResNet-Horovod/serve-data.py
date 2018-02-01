#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serve-data.py

import argparse
import os
import multiprocessing as mp

from tensorpack.dataflow import send_dataflow_zmq, MapData, TestDataSpeed
from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow)

from zmq_ops import dump_arrays


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors, parallel=min(50, mp.cpu_count()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--batch', help='per-GPU batch size',
                        default=64, type=int)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    ds = get_data('train', args.batch)

    if args.benchmark:
        ds = MapData(ds, dump_arrays)
        TestDataSpeed(ds, warmup=300).start()
    else:
        send_dataflow_zmq(
            ds, 'ipc://@imagenet-train-b{}'.format(args.batch),
            hwm=150, format='zmq_ops', bind=True)

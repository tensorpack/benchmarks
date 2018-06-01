#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-dataflow.py

import argparse
import cv2
import numpy as np
import os

from tensorpack import *
from tensorpack.dataflow.imgaug import *
from tensorpack.dataflow.parallel import PlasmaGetData, PlasmaPutData
from tensorpack.utils.serialize import loads
from tensorpack.dataflow.dftools import *

import augmentors

def test_orig(dir, name, augs, batch):
    ds = dataset.ILSVRC12(dir, name, shuffle=True)
    ds = AugmentImageComponent(ds, augs)

    ds = BatchData(ds, batch)
    #ds = PlasmaPutData(ds)
    ds = PrefetchDataZMQ(ds, 50, hwm=80)
    #ds = PlasmaGetData(ds)
    return ds

def test_lmdb_train(db, augs, batch):
    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = PrefetchData(ds, 5000, 1)
    return ds

    ds = LMDBDataPoint(ds)

    def f(x):
        return cv2.imdecode(x, cv2.IMREAD_COLOR)
    ds = MapDataComponent(ds, f, 0)
    ds = AugmentImageComponent(ds, augs)

    ds = BatchData(ds, batch, use_list=True)
    #ds = PlasmaPutData(ds)
    ds = PrefetchDataZMQ(ds, 40, hwm=80)
    #ds = PlasmaGetData(ds)
    return ds

def test_lmdb_inference(db, augs, batch):
    ds = LMDBData(db, shuffle=False)
    # ds = LocallyShuffleData(ds, 50000)

    augs = AugmentorList(augs)
    def mapper(data):
        im, label = loads(data[1])
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = augs.augment(im)
        return im, label

    ds = MultiProcessMapData(ds, 40, mapper,
            buffer_size=200)
    #ds = MultiThreadMapData(ds, 40, mapper,
            #buffer_size=2000)

    ds = BatchData(ds, batch)
    ds = PrefetchDataZMQ(ds, 1)
    return ds


def test_inference(dir, name, augs, batch=128):
    ds = dataset.ILSVRC12Files(dir, name, shuffle=False, dir_structure='train')

    aug = imgaug.AugmentorList(augs)
    def mapf(dp):
        fname, cls = dp
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = aug.augment(im)
        return im, cls
    ds = ThreadedMapData(ds, 30, mapf, buffer_size=2000, strict=True)
    ds = BatchData(ds, batch)
    ds = PrefetchDataZMQ(ds, 1)
    return ds


if __name__ == '__main__':
    available_augmentors = [
        k[:-len("_augmentor")]
        for k in augmentors.__all__ if k.endswith('_augmentor')]
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='file or directory of dataset')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--name', choices=['train', 'val'], default='train')
    parser.add_argument('--aug', choices=available_augmentors, required=True)
    args = parser.parse_args()

    augs = getattr(augmentors, args.aug + '_augmentor')()

    if args.data.endswith('lmdb'):
        if args.name == 'train':
            ds = test_lmdb_train(args.data, augs, args.batch)
        else:
            ds = test_lmdb_inference(args.data, augs, args.batch)
    else:
        if args.name == 'train':
            ds = test_orig(args.data, args.name, augs, args.batch)
        else:
            ds = test_inference(args.data, args.name, augs, args.batch)
    TestDataSpeed(ds, 500000, warmup=100).start()

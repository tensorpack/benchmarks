#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-ImageNet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import numpy as np
import os
from tensorpack import *
from tensorpack.utils.serialize import loads
from tensorpack.dataflow.dftools import *

augmentors_small = [imgaug.Resize(256)]


class Resize(imgaug.ImageAugmentor):
    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(0.08, 1.0) * area
            aspectR = self.rng.uniform(0.75, 1.333)
            ww = int(np.sqrt(targetArea * aspectR))
            hh = int(np.sqrt(targetArea / aspectR))
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                return out
        out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return out

augmentors_large = [
    Resize(),
    imgaug.RandomOrderAug(
        [imgaug.Brightness(30, clip=False),
         imgaug.Contrast((0.8, 1.2), clip=False),
         imgaug.Saturation(0.4),
         imgaug.Lighting(0.1,
                         eigval=[0.2175, 0.0188, 0.0045],
                         eigvec=[[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]]
                         )]),
    imgaug.Clip(),
    imgaug.Flip(horiz=True),
    imgaug.ToUint8()
]

def dump(dir, name, db):
    class RawILSVRC12(DataFlow):
        def __init__(self):
            meta = dataset.ILSVRCMeta()
            self.imglist = meta.get_image_list(name, dir_structure='train')
            np.random.shuffle(self.imglist)
            self.dir = os.path.join(dir, name)
        def get_data(self):
            for fname, label in self.imglist:
                fname = os.path.join(self.dir, fname)
                im = cv2.imread(fname)
                if im is None:
                    print(fname)
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                assert len(jpeg) > 10
                yield jpeg, label
        def size(self):
            return len(self.imglist)

    ds = RawILSVRC12()
    ds = PrefetchDataZMQ(ds, 1)
    dump_dataflow_to_lmdb(ds, db)

def test_orig(dir, name, augs, batch=256):
    ds = dataset.ILSVRC12(dir, name, shuffle=True)
    ds = AugmentImageComponent(ds, augs)

    ds = PrefetchDataZMQ(ds, 30)
    ds = BatchData(ds, batch)
    ds = TestDataSpeed(ds, 20000)
    ds.start_test()

def test_lmdb(db, augs, batch=256):
    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = PrefetchData(ds, 5000, 1)

    ds = LMDBDataPoint(ds)
    def f(x):
        try:
            return cv2.imdecode(x, cv2.IMREAD_COLOR)
        except:
            print(x)
            raise
    ds = MapDataComponent(ds, f, 0)
    ds = AugmentImageComponent(ds, augs)

    ds = PrefetchDataZMQ(ds, 40)
    ds = BatchData(ds, batch)
    TestDataSpeed(ds, 500000).start_test()


#dump('/datasets01/imagenet_full_size/061417', 'val',
        #'/scratch/yuxinwu/Imagenet-Val.lmdb')
test_orig('/datasets01/imagenet_full_size/061417', 'train', augmentors_large)
#test_lmdb('/checkpoint/yuxinwu/data/Imagenet-Val.lmdb', augmentors_small)
#test_lmdb('/scratch/yuxinwu/Imagenet-Val.lmdb', augmentors_small)

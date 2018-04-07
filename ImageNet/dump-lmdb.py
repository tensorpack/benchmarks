#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-lmdb.py

import numpy as np
import cv2
import os
import argparse

from tensorpack.dataflow import *

class RawILSVRC12(DataFlow):
    def __init__(self, dir, name):
        self.dir = os.path.join(dir, name)

        meta = dataset.ILSVRCMeta()
        self.imglist = meta.get_image_list(
            name,
            dataset.ILSVRCMeta.guess_dir_structure(self.dir))
        np.random.shuffle(self.imglist)

    def get_data(self):
        for fname, label in self.imglist:
            fname = os.path.join(self.dir, fname)
            im = cv2.imread(fname)
            assert im is not None, fname
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            assert len(jpeg) > 10
            yield [jpeg, label]

    def size(self):
        return len(self.imglist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to ILSVRC12 images')
    parser.add_argument('--name', choices=['train', 'val'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.output.endswith('.lmdb')

    ds = RawILSVRC12(args.data, args.name)
    ds = PrefetchDataZMQ(ds, 1)
    dftools.dump_dataflow_to_lmdb(ds, args.output)

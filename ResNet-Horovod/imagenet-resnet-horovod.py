#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet-horovod.py

import argparse
import os
import multiprocessing as mp
import cv2

from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

import horovod.tensorflow as hvd

from imagenet_utils import (
    fbresnet_augmentor, ImageNetModel, eval_on_ILSVRC12)
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)


"""
Run on single machine:

    python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    mpirun -np 8 --output-filename test.log python3 ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64

Run on multiple machines:
    host1: python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    host2: python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    mpirun -np 16 -H host1:8,host2:8 --output-filename test.log
        -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH python3
        ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64

    MPI does not like fork(), so running `serve-data.py` inside MPI is not a
    good idea. It actually also makes my data slow, which I don't know why.
    Maybe something specific to my environments.

Benchmark data:

    python3 ./serve-data.py --data ~/data/imagenet/ --batch 64 --benchmark
    # Image/s = itr/s * 64

Benchmark training:
    Train with `--fake`.

Performance on V100s:
    1 machine: 2440 im/s
    2 machine, fake data: 2381 * 2 im/s
    2 machine, true data: 2291 * 2 im/s

Note:
    For performance, the epoch length of the master process is the right number to look at.
    Epoch length of workers will be longer because they wait for master to run
    the unnecessary callbacks (save model, evaluation).

    Sometimes MPI fails to terminate all processes.
"""


class Model(ImageNetModel):
    def __init__(self, depth, data_format='NCHW', mode='resnet'):
        super(Model, self).__init__(data_format)

        if mode == 'se':
            assert depth >= 50

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck), 101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)


def get_val_data(batch):
    augmentors = fbresnet_augmentor(False)

    ds = dataset.ILSVRC12Files(args.data, 'val', shuffle=False)
    aug = imgaug.AugmentorList(augmentors)

    def mapf(dp):
        fname, cls = dp
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = aug.augment(im)
        return im, cls

    ds = MultiThreadMapData(ds, min(40, mp.cpu_count()), mapf, buffer_size=2000, strict=True)
    ds = BatchData(ds, batch, remainder=True)
    # do not fork() under MPI
    return ds


def get_config(model, fake=False):
    batch = args.batch
    total_batch = batch * hvd.size()

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        data = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        data = StagingInput(QueueInput(data))
        callbacks = []
    else:
        logger.info("#Tower: {}; Batch size per tower: {}".format(hvd.size(), batch))
        data = ZMQInput('ipc://@imagenet-train-b{}'.format(batch), 30, bind=False)
        data = StagingInput(data)

        BASE_LR = 0.1 * (total_batch // 256)
        logger.info("Base LR: {}".format(BASE_LR))
        callbacks = [
            ModelSaver(),
            ScheduledHyperParamSetter(
                'learning_rate', [(30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2),
                                  (85, BASE_LR * 1e-3), (95, BASE_LR * 1e-4),
                                  (105, BASE_LR * 1e-5)]),
        ]
        if BASE_LR != 0.1:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, 0.1), (3, BASE_LR)],
                    interp='linear'))

        # For distributed training, you probably don't want everyone to wait for validation.
        # Better to start a separate job, since the model is saved.
        if hvd.rank() == 0:
            # for reproducibility, do not use remote data for validation
            dataset_val = get_val_data(batch)
            infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                    ClassificationError('wrong-top5', 'val-error-top5')]
            # now it fails: https://github.com/uber/horovod/issues/159
            #callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1280000 // total_batch,
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch', help='per-GPU batch size', default=64, type=int)
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.data_format, args.mode)
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_val_data(batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        hvd.init()
        if hvd.rank() == 0:
            if args.fake:
                logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
            else:
                logger.set_logger_dir(
                    os.path.join(
                        'train_log',
                        'imagenet-{}-d{}'.format(args.mode, args.depth)), 'd')

        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = HorovodTrainer()
        launch_train_with_config(config, trainer)

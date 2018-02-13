#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet-horovod.py

import argparse
import os
import socket
import numpy as np
import multiprocessing as mp
import cv2

from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

import horovod.tensorflow as hvd

from imagenet_utils import (
    fbresnet_augmentor, ImageNetModel, eval_on_ILSVRC12)
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone)


"""
Run on single machine:

    python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    mpirun -np 8 --output-filename test.log python3 ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64

Run on multiple machines with RoCE/IB:
    host1: python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    host2: python3 ./serve-data.py --data ~/data/imagenet/ --batch 64
    mpirun -np 16 -H host1:8,host2:8 --output-filename test.log \
        -bind-to none -map-by slot \
        -mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
        -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH python3 -x NCCL_DEBUG=INFO \
        ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64

    MPI does not like fork(), so running `serve-data.py` inside MPI is not a
    good idea. It actually also makes my data slow, which I don't know why.
    Maybe something specific to my environments.

    Remove some MPI arguments if running with plain TCP.
    See https://github.com/uber/horovod/blob/master/docs/benchmarks.md for details.

Benchmark data:

    python3 ./serve-data.py --data ~/data/imagenet/ --batch 64 --benchmark
    # image/s = itr/s * 64

Benchmark training:
    Train with `--fake`.

Performance on V100s (batch 64):
    1 machine fake data: 2400 im/s
    2 machine, fake data: 2381 * 2 im/s
    2 machine, true data: 2291 * 2 im/s

Performanec on P100s (batch 64):
    1 machine fake data: 1638 im/s
    16 machine, fake data: 1489 * 16im/s
    16 machine, true data: 1464 * 16im/s

Note:
    For speed measurement, the epoch length of the master process is the right number to look at.
    Epoch length of workers will be longer because they wait for master to run
    the unnecessary callbacks (save model, evaluation).

    Sometimes MPI fails to terminate all processes.
"""


class Model(ImageNetModel):
    def __init__(self, depth, loss_scale=1.0):
        super(Model, self).__init__('NCHW')
        self._loss_scale = loss_scale
        self.num_blocks, self.block_func = {
            50: ([3, 4, 6, 3], resnet_bottleneck),
            101: ([3, 4, 23, 3], resnet_bottleneck),
            152: ([3, 8, 36, 3], resnet_bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            return resnet_backbone(image, self.num_blocks, resnet_group, self.block_func)

    def _build_graph(self, inputs):
        """
        Sec 3: Remark 3: Normalize the per-worker loss by
        total minibatch size kn, not per-worker size n.
        """
        super(Model, self)._build_graph(inputs)
        if self._loss_scale != 1.0:
            self.cost = self.cost * self._loss_scale

    # TODO Sec 3: momentum correction


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
            [[args.batch, 224, 224, 3], [args.batch]], 1000,
            random=False, dtype=['uint8', 'int32'])
        data = StagingInput(QueueInput(data))
        callbacks = []
        steps_per_epoch = 50
    else:
        logger.info("#Tower: {}; Batch size per tower: {}".format(hvd.size(), batch))
        data = ZMQInput('ipc://@imagenet-train-b{}'.format(batch), 30, bind=False)
        #data = StagingInput(data, nr_stage=2, device='/cpu:0')
        data = StagingInput(data, nr_stage=1)

        steps_per_epoch = int(np.round(1281167 / total_batch))

        """
        Sec 2.1: Linear Scaling Rule: When the minibatch size is multiplied by k, multiply the learning rate by k.
        """
        BASE_LR = 0.1 * (total_batch // 256)
        logger.info("Base LR: {}".format(BASE_LR))
        """
        Sec 5.1:
        We call this number (0.1 * kn / 256 ) the reference learning rate,
        and reduce it by 1/10 at the 30-th, 60-th, and 80-th epoch
        """
        callbacks = [
            ModelSaver(max_to_keep=100),
            ScheduledHyperParamSetter(
                'learning_rate', [(30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2),
                                  (80, BASE_LR * 1e-3)]),
        ]
        if BASE_LR != 0.1:
            """
            Sec 2.2: In practice, with a large minibatch of size kn, we start from a learning rate of η and increment
            it by a constant amount at each iteration such that it reachesη = kη after 5 epochs. After the warmup phase, we go back
            to the original learning rate schedule.
            """
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, 0.1), (5 * steps_per_epoch, BASE_LR)],
                    interp='linear', step_based=True))

        if hvd.rank() == 0:
            #callbacks.append(GPUUtilizationTracker())

            # TODO For distributed training, you probably don't want everyone to wait for validation.
            # Better to start a separate job, since the model is saved.
            # For reproducibility, do not use remote data for validation
            dataset_val = get_val_data(batch)
            infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                    ClassificationError('wrong-top5', 'val-error-top5')]
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=35 if args.fake else 90,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--logdir', default='train_log/tmp')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    """
    Sec 2.3: We keep the per-worker sample size n constant when we change the number of workers k.
    In this work, we use n = 32 which has performed well for a wide range of datasets and networks.
    """
    parser.add_argument('--batch', help='per-GPU batch size', default=32, type=int)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.info("Running on {}".format(socket.gethostname()))

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_val_data(batch)
        model = Model(args.depth)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        hvd.init()
        if hvd.rank() == 0:
            logger.set_logger_dir(args.logdir, 'd')

        model = Model(args.depth, loss_scale=1.0 / hvd.size())
        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        """
        Sec 3: standard communication primitives like
        allreduce [11] perform summing, not averaging
        """
        trainer = HorovodTrainer(average=False)
        launch_train_with_config(config, trainer)

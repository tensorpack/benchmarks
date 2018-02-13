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
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel, eval_on_ILSVRC12)
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone)


class Model(ImageNetModel):
    def __init__(self, depth, loss_scale=1.0):
        super(Model, self).__init__('NCHW')
        self._loss_scale = loss_scale
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            return resnet_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck)

    def _build_graph(self, inputs):
        """
        Sec 3: Remark 3: Normalize the per-worker loss by
        total minibatch size kn, not per-worker size n.
        """
        super(Model, self)._build_graph(inputs)
        if self._loss_scale != 1.0:
            self.cost = self.cost * self._loss_scale

    # TODO Sec 3: momentum correction


def get_config(model, fake=False):
    batch = args.batch
    total_batch = batch * hvd.size()

    if fake:
        data = FakeData(
            [[args.batch, 224, 224, 3], [args.batch]], 1000,
            random=False, dtype=['uint8', 'int32'])
        data = StagingInput(QueueInput(data))
        callbacks = []
        steps_per_epoch = 50
    else:
        logger.info("#Tower: {}; Batch size per tower: {}".format(hvd.size(), batch))
        data = ZMQInput('ipc://@imagenet-train-b{}'.format(batch), 30, bind=False)
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

    if hvd.rank() == 0 and not args.fake:
        # TODO For distributed training, you probably don't want everyone to wait for validation.
        # Better to start a separate job, since the model is saved.
        if False:
            dataset_val = get_imagenet_dataflow(
                args.data, 'val', batch, fbresnet_augmentor(False))  # For reproducibility, do not use remote data for validation
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
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--logdir', default='train_log/tmp')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    """
    Sec 2.3: We keep the per-worker sample size n constant when we change the number of workers k.
    In this work, we use n = 32 which has performed well for a wide range of datasets and networks.
    """
    parser.add_argument('--batch', help='per-GPU batch size', default=32, type=int)
    args = parser.parse_args()

    logger.info("Running on {}".format(socket.gethostname()))

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_val_data(batch)
        model = Model(args.depth)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        assert args.load is None
        hvd.init()
        if hvd.rank() == 0:
            logger.set_logger_dir(args.logdir, 'd')

        model = Model(args.depth, loss_scale=1.0 / hvd.size())
        config = get_config(model, fake=args.fake)
        """
        Sec 3: standard communication primitives like
        allreduce [11] perform summing, not averaging
        """
        trainer = HorovodTrainer(average=False)
        launch_train_with_config(config, trainer)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet-horovod.py

import argparse
import sys
import os
import socket
import numpy as np

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader

import horovod.tensorflow as hvd

from imagenet_utils import (
    fbresnet_augmentor, get_val_dataflow, ImageNetModel, eval_classification)
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone,
    weight_standardization_context, Norm)


class Model(ImageNetModel):
    def __init__(self, depth, norm='BN', use_ws=False):
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        self.norm = norm
        self.use_ws = use_ws

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
                argscope(Norm, type=self.norm), \
                weight_standardization_context(enable=self.use_ws):
            return resnet_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck)


class HorovodClassificationError(ClassificationError):
    """
    Like ClassificationError, it evaluates total samples & count of wrong samples.
    But in the end we aggregate the total&count by horovod.
    """
    def _setup_graph(self):
        self._placeholder = tf.placeholder(tf.float32, shape=[2], name='to_be_reduced')
        self._reduced = hvd.allreduce(self._placeholder, average=False)

    def _after_inference(self):
        tot = self.err_stat.total
        cnt = self.err_stat.count
        tot, cnt = self._reduced.eval(feed_dict={self._placeholder: [tot, cnt]})
        return {self.summary_name: cnt * 1. / tot}


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
        zmq_addr = 'ipc://@imagenet-train-b{}'.format(batch)
        if args.no_zmq_ops:
            dataflow = RemoteDataZMQ(zmq_addr, hwm=150, bind=False)
            data = QueueInput(dataflow)
        else:
            data = ZMQInput(zmq_addr, 30, bind=False)
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
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter(
            'learning_rate', [(0, BASE_LR), (30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2),
                              (80, BASE_LR * 1e-3)]),
    ]
    if BASE_LR > 0.1:
        """
        Sec 2.2: In practice, with a large minibatch of size kn, we start from a learning rate of η and increment
        it by a constant amount at each iteration such that it reachesη = kη after 5 epochs.
        After the warmup phase, we go back to the original learning rate schedule.
        """
        callbacks.append(
            ScheduledHyperParamSetter(
                'learning_rate', [(0, 0.1), (5 * steps_per_epoch, BASE_LR)],
                interp='linear', step_based=True))

    if args.validation is not None:
        # TODO For distributed training, you probably don't want everyone to wait for master doing validation.
        # Better to start a separate job, since the model is saved.
        if args.validation == 'master' and hvd.rank() == 0:
            # For reproducibility, do not use remote data for validation
            dataset_val = get_val_dataflow(
                args.data, 64, fbresnet_augmentor(False))
            infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                    ClassificationError('wrong-top5', 'val-error-top5')]
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        # For simple validation tasks such as image classification, distributed validation is possible.
        elif args.validation == 'distributed':
            dataset_val = get_val_dataflow(
                args.data, 64, fbresnet_augmentor(False),
                num_splits=hvd.size(), split_index=hvd.rank())
            infs = [HorovodClassificationError('wrong-top1', 'val-error-top1'),
                    HorovodClassificationError('wrong-top5', 'val-error-top5')]
            callbacks.append(
                InferenceRunner(QueueInput(dataset_val), infs).set_chief_only(False))

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=35 if args.fake else 95,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--logdir', help='Directory for models and training stats.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', action='store_true', help='run evaluation with --load instead of training.')

    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[50, 101, 152])
    parser.add_argument('--use-ws', action='store_true')
    parser.add_argument('--norm', choices=['BN', 'GN'], default='BN')
    parser.add_argument('--accum-grad', type=int, default=1)
    parser.add_argument('--weight-decay-norm', action='store_true',
                        help="apply weight decay on normalization layers (gamma & beta)."
                             "This is used in torch/pytorch, and slightly "
                             "improves validation accuracy of large models.")
    parser.add_argument('--validation', choices=['distributed', 'master'],
                        help='Validation method. By default the script performs no validation.')
    parser.add_argument('--no-zmq-ops', help='use pure python to send/receive data',
                        action='store_true')
    """
    Sec 2.3: We keep the per-worker sample size n constant when we change the number of workers k.
    In this work, we use n = 32 which has performed well for a wide range of datasets and networks.
    """
    parser.add_argument('--batch', help='per-GPU batch size', default=32, type=int)
    args = parser.parse_args()

    model = Model(args.depth, args.norm, args.use_ws)
    model.accum_grad = args.accum_grad
    if args.weight_decay_norm:
        model.weight_decay_pattern = ".*/W|.*/gamma|.*/beta"""

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_val_dataflow(args.data, batch, fbresnet_augmentor(False))
        eval_classification(model, get_model_loader(args.load), ds)
        sys.exit()

    logger.info("Training on {}".format(socket.gethostname()))
    # Print some information for sanity check.
    os.system("nvidia-smi")
    assert args.load is None

    hvd.init()

    if args.logdir is None:
        args.logdir = os.path.join('train_log', 'Horovod-{}GPUs-{}Batch'.format(hvd.size(), args.batch))

    if hvd.rank() == 0:
        logger.set_logger_dir(args.logdir, 'd')
    logger.info("Rank={}, Local Rank={}, Size={}".format(hvd.rank(), hvd.local_rank(), hvd.size()))

    """
    Sec 3: Remark 3: Normalize the per-worker loss by
    total minibatch size kn, not per-worker size n.
    """
    model.loss_scale = 1.0 / hvd.size()
    config = get_config(model, fake=args.fake)
    """
    Sec 3: standard communication primitives like
    allreduce [11] perform summing, not averaging
    """
    trainer = HorovodTrainer(average=False)
    launch_train_with_config(config, trainer)

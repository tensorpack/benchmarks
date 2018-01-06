#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: resnet-multigpu.py

import sys
import argparse
import numpy as np
import os
from contextlib import contextmanager
import tensorflow as tf

from tensorpack import *
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.utils.stats import RatioCounter
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.collection import freeze_collection
from tensorpack.tfutils import get_current_tower_context

from tfbench.convnet_builder import ConvNetBuilder
from tfbench import model_config

INPUT_SHAPE = 224
IMAGE_DTYPE = tf.float32
IMAGE_DTYPE_NUMPY = 'float32'


class Model(ModelDesc):
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(IMAGE_DTYPE, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        return tf.train.GradientDescentOptimizer(lr)

    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        image, label = inputs

        # all-zero tensor hurt performance for some reason.
        label = tf.random_uniform(
            [64],
            minval=0, maxval=1000 - 1,
            dtype=tf.int32, name='synthetic_labels')

        # our fake images are in [0, 1]
        image = tf.cast(image, tf.float32) * 2.0 - 1.
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self._get_logits(image)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')
        if False:
            self.cost = loss    # disable wd
        else:
            wd_cost = regularize_cost('.*', tf.nn.l2_loss) * 1e-4
            self.cost = tf.add_n([loss, wd_cost], name='cost')

@contextmanager
def maybe_freeze_updates(enable):
    if enable:
        with freeze_collection([tf.GraphKeys.UPDATE_OPS]):
            yield
    else:
        yield

class TFBenchModel(Model):
    def _get_logits(self, image):
        ctx = get_current_tower_context()

        with maybe_freeze_updates(ctx.index > 0):
            network = ConvNetBuilder(
                image, 3, True, True, data_format=self.data_format,
                dtype=tf.float32, variable_dtype=tf.float32)
            dataset = lambda: 1
            dataset.name = 'imagenet'
            model_conf = model_config.get_model_config('resnet50', dataset)
            model_conf.set_batch_size(64)
            model_conf.add_inference(network)
            return network.affine(1000, activation='linear', stddev=0.001)


class TensorpackModel(Model):
    """
    Implement the same model with tensorpack layers.
    """
    def _get_logits(self, image):
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
                l = BatchNorm('bnshortcut', l)
                return l
            else:
                return l

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            input = l
            l = Conv2D('conv1', l, ch_out, 1, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=1, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1, nl=tf.identity)
            l = BatchNorm('bn', l)
            ret = l + shortcut(input, ch_in, ch_out * 4, stride)
            return tf.nn.relu(ret)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        defs = [3, 4, 6, 3]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', bottleneck, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', bottleneck, 128, defs[1], 2)
                      .apply(layer, 'group2', bottleneck, 256, defs[2], 2)
                      .apply(layer, 'group3', bottleneck, 512, defs[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())
        return logits


def get_data(mode):
    # get input
    input_shape = [64, 224, 224, 3]
    label_shape = [64]
    dataflow = FakeData(
        [input_shape, label_shape], 1000,
        random=False, dtype=[IMAGE_DTYPE_NUMPY, 'int32'])
    if mode == 'gpu':
        return DummyConstantInput([input_shape, label_shape])
    elif mode == 'cpu':
        def fn():
            # these copied from tensorflow/benchmarks
            with tf.device('/cpu:0'):
                images = tf.truncated_normal(
                    input_shape, dtype=IMAGE_DTYPE, stddev=1e-1, name='synthetic_images')
                labels = tf.random_uniform(
                    label_shape, minval=1, maxval=1000, dtype=tf.int32, name='synthetic_labels')
                # images = tf.contrib.framework.local_variable(images, name='images')
            return [images, labels]
        ret = TensorInput(fn)
        return StagingInput(ret, nr_stage=1)
    elif mode == 'python':
        # try the speed of dataset as well.
        # ds = TFDatasetInput.dataflow_to_dataset(dataflow, [IMAGE_DTYPE, tf.int32])
        # ds = ds.prefetch(30)
        # ret = TFDatasetInput(ds)
        ret = QueueInput(
            dataflow,
            queue=tf.FIFOQueue(100, [tf.uint8, tf.int32]))
        return StagingInput(ret, nr_stage=1)
    elif mode == 'zmq-serve':
        send_dataflow_zmq(dataflow, 'ipc://testpipe', hwm=100, format='zmq_op')
        sys.exit()
    elif mode == 'zmq-consume':
        ret = ZMQInput(
            'ipc://testpipe', hwm=100)
        return StagingInput(ret, nr_stage=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', choices=['tfbench', 'tensorpack'], default='tfbench')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('--fake-location', help='the place to create fake data',
                        type=str, default='gpu', choices=['cpu', 'gpu', 'python', 'zmq-serve', 'zmq-consume'])
    parser.add_argument('--variable-update', help='variable update strategy',
                        type=str,
                        choices=['replicated', 'parameter_server', 'horovod'],
                        required=True)

    parser.add_argument('--ps-hosts')
    parser.add_argument('--worker-hosts')
    parser.add_argument('--job')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sessconf = get_default_sess_config()
    sessconf.inter_op_parallelism_threads = 80 - 16
    if args.job:
        # distributed:
        cluster_spec = tf.train.ClusterSpec({
            'ps': args.ps_hosts.split(','),
            'worker': args.worker_hosts.split(',')
        })
        job = args.job.split(':')[0]
        if job == 'ps':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        task_index = int(args.job.split(':')[1])
        server = tf.train.Server(
            cluster_spec, job_name=job, task_index=task_index,
            config=sessconf)

    NR_GPU = get_nr_gpu()

    if args.job:
        trainer = {
            'replicated': lambda: DistributedTrainerReplicated(NR_GPU, server),
            'parameter_server': lambda: DistributedTrainerParameterServer(NR_GPU, server),
        }[args.variable_update]()
    else:
        if NR_GPU == 1:
            trainer = SimpleTrainer()
        else:
            trainer = {
                'replicated': lambda: SyncMultiGPUTrainerReplicated(NR_GPU),
                'horovod': lambda: HorovodTrainer(),
                'parameter_server': lambda: SyncMultiGPUTrainerParameterServer(NR_GPU, ps_device='cpu')
            }[args.variable_update]()

    M = TFBenchModel if args.model == 'tfbench' else TensorpackModel
    config = TrainConfig(
        data=get_data(args.fake_location),
        model=M(data_format=args.data_format),
        callbacks=[
            GPUUtilizationTracker(),
            # ModelSaver(checkpoint_dir='./tmpmodel'),
        ],
        extra_callbacks=[
            # MovingAverageSummary(),
            ProgressBar(),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        session_config=sessconf,
        steps_per_epoch=100,
        max_epoch=10,
    )


    # consistent with tensorflow/benchmarks
    trainer.COLOCATE_GRADIENTS_WITH_OPS = False
    launch_train_with_config(config, trainer)

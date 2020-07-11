#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet-multigpu.py

import sys
import argparse
import os
from contextlib import contextmanager, ExitStack
import tensorflow as tf

from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import *
from tensorpack.utils.argtools import log_once

from tensorpack.tfutils.collection import freeze_collection
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.varreplace import custom_getter_scope

from tfbench.convnet_builder import ConvNetBuilder
from tfbench import model_config

INPUT_SHAPE = 224
IMAGE_DTYPE = tf.uint8
IMAGE_DTYPE_NUMPY = 'uint8'


class Model(ModelDesc):
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def inputs(self):
        return [tf.TensorSpec([args.batch, INPUT_SHAPE, INPUT_SHAPE, 3], IMAGE_DTYPE, 'input'),
                tf.TensorSpec([args.batch], tf.int32, 'label')]

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        return tf.train.GradientDescentOptimizer(lr)

    def build_graph(self, image, label):
        # all-zero tensor hurt performance for some reason.
        ctx = get_current_tower_context()
        label = tf.random_uniform(
            [args.batch],
            minval=0, maxval=1000 - 1,
            dtype=tf.int32, name='synthetic_labels')

        # our fake images are in [0, 1]
        target_dtype = tf.float16 if args.use_fp16 else tf.float32
        if image.dtype != target_dtype:
            image = tf.cast(image, target_dtype) * 2.0 - 1.
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self._get_logits(image)

        if logits.dtype != tf.float32:
            logger.info("Casting logits back to fp32 ...")
            logits = tf.cast(logits, tf.float32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')
        if ctx.index == ctx.total - 1:
            wd_cost = regularize_cost('.*', tf.nn.l2_loss) * (1e-4 * ctx.total)
            return tf.add_n([loss, wd_cost], name='cost')
        else:
            return loss


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
                image, 3, True,
                use_tf_layers=True,
                data_format=self.data_format,
                dtype=tf.float16 if args.use_fp16 else tf.float32,
                variable_dtype=tf.float32)
            with custom_getter_scope(network.get_custom_getter()):
                dataset = lambda: 1
                dataset.name = 'imagenet'
                model_conf = model_config.get_model_config('resnet50', dataset)
                model_conf.set_batch_size(args.batch)
                model_conf.add_inference(network)
                return network.affine(1000, activation='linear', stddev=0.001)


class TensorpackModel(Model):
    """
    Implement the same model with tensorpack layers.
    """
    def _get_logits(self, image):

        def fp16_getter(getter, *args, **kwargs):
            name = args[0] if len(args) else kwargs['name']
            if not name.endswith('/W') and not name.endswith('/b'):
                """
                Following convention, convolution & fc are quantized.
                BatchNorm (gamma & beta) are not quantized.
                """
                return getter(*args, **kwargs)
            else:
                if kwargs['dtype'] == tf.float16:
                    kwargs['dtype'] = tf.float32
                    ret = getter(*args, **kwargs)
                    ret = tf.cast(ret, tf.float16)
                    log_once("Variable {} casted to fp16 ...".format(name))
                    return ret
                else:
                    return getter(*args, **kwargs)

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, strides=stride)
                l = BatchNorm('bnshortcut', l)
                return l
            else:
                return l

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            input = l
            l = Conv2D('conv1', l, ch_out, 1, strides=stride, activation=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, strides=1, activation=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1, activation=tf.identity)
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

        with ExitStack() as stack:
            stack.enter_context(argscope(
                Conv2D, use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(mode='fan_out')))
            stack.enter_context(argscope(
                [Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
                data_format=self.data_format))
            if args.use_fp16:
                stack.enter_context(custom_getter_scope(fp16_getter))
                image = tf.cast(image, tf.float16)
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, strides=2)
                      .BatchNorm('bn0')
                      .tf.nn.relu()
                      .MaxPooling('pool0', 3, strides=2, padding='SAME')
                      .apply(layer, 'group0', bottleneck, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', bottleneck, 128, defs[1], 2)
                      .apply(layer, 'group2', bottleneck, 256, defs[2], 2)
                      .apply(layer, 'group3', bottleneck, 512, defs[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000)())
        if args.use_fp16:
            logits = tf.cast(logits, tf.float32)
        return logits


def get_data(mode):
    # get input
    input_shape = [args.batch, 224, 224, 3]
    label_shape = [args.batch]
    dataflow = FakeData(
        [input_shape, label_shape], 1000,
        random=False, dtype=[IMAGE_DTYPE_NUMPY, 'int32'])
    if mode == 'gpu':
        return DummyConstantInput([input_shape, label_shape])
    elif mode == 'cpu':
        def fn():
            # these copied from tensorflow/benchmarks
            with tf.device('/cpu:0'):
                if IMAGE_DTYPE == tf.float32:
                    images = tf.truncated_normal(
                        input_shape, dtype=IMAGE_DTYPE, stddev=0.1, name='synthetic_images')
                else:
                    images = tf.random_uniform(
                        input_shape, minval=0, maxval=255, dtype=tf.int32, name='synthetic_images')
                    images = tf.cast(images, IMAGE_DTYPE)
                labels = tf.random_uniform(
                    label_shape, minval=1, maxval=1000, dtype=tf.int32, name='synthetic_labels')
                # images = tf.contrib.framework.local_variable(images, name='images')
            return [images, labels]
        ret = TensorInput(fn)
        return StagingInput(ret, nr_stage=1)
    elif mode == 'python' or mode == 'python-queue':
        ret = QueueInput(
            dataflow,
            queue=tf.FIFOQueue(args.prefetch, [IMAGE_DTYPE, tf.int32]))
        return StagingInput(ret, nr_stage=1)
    elif mode == 'python-dataset':
        ds = TFDatasetInput.dataflow_to_dataset(dataflow, [IMAGE_DTYPE, tf.int32])
        ds = ds.repeat().prefetch(args.prefetch)
        ret = TFDatasetInput(ds)
        return StagingInput(ret, nr_stage=1)
    elif mode == 'zmq-serve':
        send_dataflow_zmq(dataflow, 'ipc://testpipe', hwm=args.prefetch, format='zmq_op')
        sys.exit()
    elif mode == 'zmq-consume':
        ret = ZMQInput(
            'ipc://testpipe', hwm=args.prefetch)
        return StagingInput(ret, nr_stage=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', choices=['tfbench', 'tensorpack'], default='tensorpack')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--prefetch', type=int, default=150)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--use-xla-compile', action='store_true')
    parser.add_argument('--batch', type=int, default=64, help='per GPU batch size')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('--fake-location', help='the place to create fake data',
                        type=str, default='gpu',
                        choices=[
                            'cpu', 'gpu',
                            'python', 'python-queue', 'python-dataset',
                            'zmq-serve', 'zmq-consume'])
    parser.add_argument('--variable-update', help='variable update strategy',
                        type=str,
                        choices=['replicated', 'parameter_server', 'horovod', 'byteps'],
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

    NUM_GPU = get_nr_gpu()
    logger.info("Number of GPUs found: " + str(NUM_GPU))

    if args.job:
        trainer = {
            'replicated': lambda: DistributedTrainerReplicated(NUM_GPU, server),
            'parameter_server': lambda: DistributedTrainerParameterServer(NUM_GPU, server),
        }[args.variable_update]()
    else:
        if NUM_GPU == 1:
            trainer = SimpleTrainer()
        else:
            trainer = {
                'replicated': lambda: SyncMultiGPUTrainerReplicated(
                    NUM_GPU, average=False, mode='hierarchical' if NUM_GPU >= 8 else 'cpu'),
                # average=False is the actual configuration used by tfbench
                'horovod': lambda: HorovodTrainer(),
                'byteps': lambda: BytePSTrainer(),
                'parameter_server': lambda: SyncMultiGPUTrainerParameterServer(NUM_GPU, ps_device='cpu')
            }[args.variable_update]()
            if args.variable_update == 'replicated':
                trainer.BROADCAST_EVERY_EPOCH = False

    M = TFBenchModel if args.model == 'tfbench' else TensorpackModel
    callbacks = [
        GPUMemoryTracker(),
        # ModelSaver(checkpoint_dir='./tmpmodel'),  # it takes time
    ]
    if args.variable_update != 'horovod':
        callbacks.append(GPUUtilizationTracker())
    if args.variable_update in ['horovod', 'byteps']:
        # disable logging if not on first rank
        if trainer.hvd.rank() != 0:
            logger.addFilter(lambda _: False)
        NUM_GPU = trainer.hvd.size()
        logger.info("Number of GPUs to use: " + str(NUM_GPU))

    try:
        from tensorpack.callbacks import ThroughputTracker
    except ImportError:
        # only available in latest tensorpack
        pass
    else:
        callbacks.append(ThroughputTracker(samples_per_step=args.batch * NUM_GPU))

    config = TrainConfig(
        data=get_data(args.fake_location),
        model=M(data_format=args.data_format),
        callbacks=callbacks,
        extra_callbacks=[
            # MovingAverageSummary(),   # tensorflow/benchmarks does not do this
            ProgressBar(),  # nor this
            # MergeAllSummaries(),
            RunUpdateOps()
        ],
        session_config=sessconf if not args.job else None,
        steps_per_epoch=50,
        max_epoch=10,
    )

    # consistent with tensorflow/benchmarks
    trainer.COLOCATE_GRADIENTS_WITH_OPS = False
    trainer.XLA_COMPILE = args.use_xla_compile
    launch_train_with_config(config, trainer)

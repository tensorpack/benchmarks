#!/usr/bin/env python

import argparse
import numpy as np
import os
import multiprocessing as mp
import tensorflow as tf

from tensorpack import *
from tensorpack.utils import logger
from tensorpack.tfutils.tower import TowerFunc
from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.dataflow import dataset


BATCH = 512
STEPS_PER_EPOCH = 50000 // 512 + 1
TOTAL_EPOCH = 24
WARMUP_EPOCH = 5.0 / 24 * TOTAL_EPOCH
USE_FP16 = True
DATA_FORMAT = "NCHW"


def get_inputs(batch):
    return [tf.TensorSpec(
                (batch, 3, 32, 32) if DATA_FORMAT == "NCHW" else (batch, 32, 32, 3),
                tf.float32, 'input'),
            tf.TensorSpec((batch, 10), tf.float32, 'label')]


def build_graph(image, label):
    if USE_FP16:
        image = tf.cast(image, tf.float16)

    def activation(x):
        return tf.nn.leaky_relu(x, alpha=0.1)

    def residual(name, x, chan):
        with tf.variable_scope(name):
            x = Conv2D('res1', x, chan, 3)
            x = BatchNorm('bn1', x)
            x = activation(x)
            x = Conv2D('res2', x, chan, 3)
            x = BatchNorm('bn2', x)
            x = activation(x)
            return x

    def fp16_getter(getter, *args, **kwargs):
        name = args[0] if len(args) else kwargs['name']
        if not USE_FP16 or (not name.endswith('/W') and not name.endswith('/b')):
            # ignore BN's gamma and beta
            return getter(*args, **kwargs)
        else:
            if kwargs['dtype'] == tf.float16:
                kwargs['dtype'] = tf.float32
                ret = getter(*args, **kwargs)
                return tf.cast(ret, tf.float16)
            else:
                return getter(*args, **kwargs)

    with custom_getter_scope(fp16_getter), \
            argscope(Conv2D, activation=tf.identity, use_bias=False), \
            argscope([Conv2D, MaxPooling, BatchNorm], data_format=DATA_FORMAT), \
            argscope(BatchNorm, momentum=0.8):

        with tf.variable_scope('prep'):
            l = Conv2D('conv', image, 64, 3)
            l = BatchNorm('bn', l)
            l = activation(l)

        with tf.variable_scope("layer1"):
            l = Conv2D('conv', l, 128, 3)
            l = MaxPooling('pool', l, 2)
            l = BatchNorm('bn', l)
            l = activation(l)
            l = l + residual('res', l, 128)

        with tf.variable_scope("layer2"):
            l = Conv2D('conv', l, 256, 3)
            l = MaxPooling('pool', l, 2)
            l = BatchNorm('bn', l)
            l = activation(l)

        with tf.variable_scope("layer3"):
            l = Conv2D('conv', l, 512, 3)
            l = MaxPooling('pool', l, 2)
            l = BatchNorm('bn', l)
            l = activation(l)
            l = l + residual('res', l, 512)

        l = tf.reduce_max(l, axis=[2, 3] if DATA_FORMAT == "NCHW" else [1, 2])
        l = FullyConnected('fc', l, 10, use_bias=False)
        logits = tf.cast(l * 0.125, tf.float32, name='logits')

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
    cost = tf.reduce_sum(cost)
    wd_cost = regularize_cost('.*', l2_regularizer(5e-4 * BATCH), name='regularize_loss')

    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(label, axis=1), name='correct')
    return tf.add_n([cost, wd_cost], name='cost')


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)

    cifar10_mean = np.asarray([0.4914, 0.4822, 0.4465], dtype="float32") * 255.
    cifar10_invstd = 1.0 / (np.asarray([0.2471, 0.2435, 0.2616], dtype="float32") * 255)

    if isTrain:
        augmentors = imgaug.AugmentorList([
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.RandomCutout(8, 8),
        ])

    def mapf(dp):
        img, label = dp
        img = (img.astype("float32") - cifar10_mean) * cifar10_invstd

        if isTrain:
            img = np.pad(img, [(4, 4), (4, 4), (0, 0)], mode='reflect')
            img = augmentors.augment(img)

            onehot = np.zeros((10, ), dtype=np.float32) + 0.2 / 9
            onehot[label] = 0.8
        else:
            onehot = np.zeros((10, ), dtype=np.float32)
            onehot[label] = 1.

        if DATA_FORMAT == "NCHW":
            img = img.transpose(2, 0, 1)
        return img, onehot

    if not isTrain:
        ds = MapData(ds, mapf)
        ds = BatchData(ds, BATCH, remainder=False)
        return ds

    ds = MultiProcessMapAndBatchDataZMQ(ds, 8, mapf, BATCH, buffer_size=20000)
    ds = RepeatedData(ds, -1)
    return ds

def run_once(result_queue):
    tf.reset_default_graph()
    trainer = SimpleTrainer()
    trainer.XLA_COMPILE = True

    with tf.device('/gpu:0'):
        gs = tf.train.get_or_create_global_step()
        gs = tf.cast(gs, tf.float32)
        # 0.0 -> 0.4 in warmup
        # 0.4 -> 0.0 in the rest epochs
        lr = tf.where(tf.greater(gs, 5 * STEPS_PER_EPOCH),
            (TOTAL_EPOCH * STEPS_PER_EPOCH - gs) / ((TOTAL_EPOCH - WARMUP_EPOCH) * STEPS_PER_EPOCH) * 0.4 / BATCH,
            gs / (WARMUP_EPOCH * STEPS_PER_EPOCH) * 0.4 / BATCH
        )

    trainer.setup_graph(
        get_inputs(BATCH),
        #StagingInput(QueueInput(get_data('train'), queue=tf.FIFOQueue(300, [tf.float32, tf.int64]))),
        #DummyConstantInput([x.shape for x in get_inputs()]),
        StagingInput(TFDatasetInput(get_data('train'))),
        build_graph,
        lambda: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    )

    trainer.train_with_defaults(
        callbacks=[
            PeriodicTrigger(
                InferenceRunner(
                    get_data('test'), ClassificationError('correct', 'val_acc'),
                    # We used static shape in training, in order to allow XLA
                    # But we want dynamic batch size for inference, therefore
                    # recreate a tower function with different input signature.
                    tower_func=TowerFunc(build_graph, get_inputs(None))
                ), every_k_epochs=TOTAL_EPOCH),
            RunUpdateOps(),
        ],
        extra_callbacks=[],  # disable all default callbacks
        monitors=[ScalarPrinter()],  # disable other default monitors
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=TOTAL_EPOCH
    )
    result_queue.put(trainer.monitors.get_latest('val_acc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--num-runs', default=1, type=int)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TF_AUTOTUNE_THRESHOLD"] = '1'

    q = mp.Queue()
    if args.num_runs == 1:
        run_once(q)
        logger.info("Val Acc: " + str(q.get()))
    else:
        val_accs = []
        for k in range(args.num_runs):
            proc = mp.Process(target=run_once, args=(q,))
            proc.start()
            val_accs.append(q.get())
            proc.join(timeout=5)
            proc.terminate()
            logger.info("Val Accs: " + str(val_accs))
        logger.info("Mean Val Acc: " + str(np.mean(val_accs)))

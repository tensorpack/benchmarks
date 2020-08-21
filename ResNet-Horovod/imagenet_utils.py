#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py

"""
This file is modified from
https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels/imagenet_utils.py
"""


import multiprocessing as mp
import numpy as np
from abc import abstractmethod
import cv2
import tensorflow as tf

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    BatchData, MultiThreadMapData, DataFromList,
    AugmentImageComponent, MultiProcessRunnerZMQ)
from tensorpack.models import regularize_cost
from tensorpack.predict import FeedfreePredictor, PredictConfig
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.optimizer import AccumGradOptimizer
from tensorpack.utils import logger
from tensorpack.utils.stats import RatioCounter

"""
====== DataFlow =======
"""


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_LINEAR
    if isTrain:
        """
        Sec 5.1:
        We use scale and aspect ratio data augmentation [35] as
        in [12]. The network input image is a 224×224 pixel random
        crop from an augmented image or its horizontal flip.
        """
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.ToFloat32(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                 imgaug.Saturation(0.4, rgb=False, clip=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.ToUint8(),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_train_dataflow(datadir, batch, augmentors=None, parallel=None):
    """
    Sec 3, Remark 4:
    Use a single random shuffling of the training data (per epoch)
    that is divided amongst all k workers.

    NOTE: Here we do not follow the paper which makes some differences.
    Here, each machine shuffles independently.
    """
    if parallel is None:
        parallel = min(50, mp.cpu_count())
    if augmentors is None:
        augmentors = fbresnet_augmentor(True)
    ds = dataset.ILSVRC12(datadir, 'train', shuffle=True)
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, batch, remainder=False)
    ds = MultiProcessRunnerZMQ(ds, parallel)
    return ds


def get_val_dataflow(
        datadir, batch_size,
        augmentors=None, parallel=None,
        num_splits=None, split_index=None):
    if augmentors is None:
        augmentors = fbresnet_augmentor(False)
    assert datadir is not None
    assert isinstance(augmentors, list)
    if parallel is None:
        parallel = min(40, mp.cpu_count())

    if num_splits is None:
        ds = dataset.ILSVRC12Files(datadir, 'val', shuffle=False)
    else:
        # shard validation data
        assert split_index < num_splits
        files = dataset.ILSVRC12Files(datadir, 'val', shuffle=False)
        files.reset_state()
        files = list(files.get_data())
        logger.info("Number of validation data = {}".format(len(files)))
        split_size = len(files) // num_splits
        start, end = split_size * split_index, split_size * (split_index + 1)
        end = min(end, len(files))
        logger.info("Local validation split = {} - {}".format(start, end))
        files = files[start: end]
        ds = DataFromList(files, shuffle=False)
    aug = imgaug.AugmentorList(augmentors)

    def mapf(dp):
        fname, cls = dp
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = aug.augment(im)
        return im, cls
    ds = MultiThreadMapData(ds, parallel, mapf,
                            buffer_size=min(2000, ds.size()), strict=True)
    ds = BatchData(ds, batch_size, remainder=True)
    # do not fork() under MPI
    return ds


def eval_classification(model, sessinit, dataflow):
    """
    Eval a classification model on the dataset. It assumes the model inputs are
    named "input" and "label", and contains "wrong-top1" and "wrong-top5" in the graph.
    """
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    acc1, acc5 = RatioCounter(), RatioCounter()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))
    for _ in tqdm.trange(dataflow.size()):
        top1, top5 = pred()
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)

    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'

    Sec 5.1: We use a weight decay λ of 0.0001 and following [16] we do not apply
    weight decay on the learnable BN coefficients
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.

    """
    Accumulate gradients across several steps (by default 1, which means no accumulation across steps).
    """
    accum_grad = 1

    def inputs(self):
        return [tf.TensorSpec([None, self.image_shape, self.image_shape, 3], self.image_dtype, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = self.image_preprocess(image)
        assert self.data_format == 'NCHW'
        image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        """
        Sec 5.1: We use Nesterov momentum with m of 0.9.

        Sec 3: momentum correction
        Tensorflow's momentum optimizer does not need correction.
        """
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        if self.accum_grad != 1:
            opt = AccumGradOptimizer(opt, self.accum_grad)
        return opt

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            """
            Sec 5.1:
            The input image is normalized by the per-color mean and
            standard deviation, as in [12]
            """
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, label_smoothing=0.):
        if label_smoothing != 0.:
            nclass = logits.shape[-1]
            label = tf.one_hot(label, nclass) if label.shape.ndims == 1 else label

        if label.shape.ndims == 1:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        else:
            loss = tf.losses.softmax_cross_entropy(
                label, logits, label_smoothing=label_smoothing,
                reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss

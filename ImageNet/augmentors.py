#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: augmentors.py

import numpy as np
import cv2
from tensorpack.dataflow import imgaug


__all__ = ['fbresnet_augmentor', 'inference_augmentor',
           'resizeAndLighting_augmentor']



def inference_augmentor():
    return [
        imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
        imgaug.CenterCrop((224, 224))
    ]


def fbresnet_augmentor():
    # assme BGR input
    augmentors = [
        imgaug.GoogleNetRandomCropAndResize(),
        imgaug.RandomOrderAug(
            [imgaug.BrightnessScale((0.6, 1.4), clip=False),
             imgaug.Contrast((0.6, 1.4), clip=False),
             imgaug.Saturation(0.4, rgb=False),
             # rgb->bgr conversion for the constants copied from fb.resnet.torch
             imgaug.Lighting(0.1,
                             eigval=np.asarray(
                                 [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                             eigvec=np.array(
                                 [[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203]],
                                 dtype='float32')[::-1, ::-1]
                             )]),
        imgaug.Flip(horiz=True),
    ]
    return augmentors


def resizeAndLighting_augmentor():
    # assme BGR input
    augmentors = [
        imgaug.GoogleNetRandomCropAndResize(),
        imgaug.Lighting(0.1,
                        eigval=np.asarray(
                            [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                        eigvec=np.array(
                            [[-0.5675, 0.7192, 0.4009],
                             [-0.5808, -0.0045, -0.8140],
                             [-0.5836, -0.6948, 0.4203]],
                            dtype='float32')[::-1, ::-1]),
        imgaug.Flip(horiz=True),
    ]
    return augmentors


def resizeOnly_augmentor():
    # assme BGR input
    augmentors = [
        imgaug.GoogleNetRandomCropAndResize(),
        imgaug.Lighting(0.1,
                        eigval=np.asarray(
                            [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                        eigvec=np.array(
                            [[-0.5675, 0.7192, 0.4009],
                             [-0.5808, -0.0045, -0.8140],
                             [-0.5836, -0.6948, 0.4203]],
                            dtype='float32')[::-1, ::-1]),
        imgaug.Flip(horiz=True),
    ]
    return augmentors

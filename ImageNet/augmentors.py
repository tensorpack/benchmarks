#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: augmentors.py

import numpy as np
import cv2
from tensorpack.dataflow import imgaug


__all__ = ['fbresnet_augmentor', 'inference_augmentor',
           'resizeAndLighting_augmentor']


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def inference_augmentor():
    return [
        imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
        imgaug.CenterCrop((224, 224))
    ]


def fbresnet_augmentor():
    # assme BGR input
    augmentors = [
        GoogleNetResize(),
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
        GoogleNetResize(),
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
        GoogleNetResize(),
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

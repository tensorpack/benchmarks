#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-opencv-resize.py


import cv2
import time
import numpy as np

"""
Some prebuild opencv is much slower than others.

On E5-2680v3, archlinux, this script prints:

0.61s for system opencv 3.4.0-2.
>5 s for anaconda opencv 3.3.1 py36h6cbbc71_1.
"""


img = (np.random.rand(256, 256, 3) * 255).astype('uint8')

start = time.time()
for k in range(1000):
    out = cv2.resize(img, (384, 384))
print(time.time() - start)

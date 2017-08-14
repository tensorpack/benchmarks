#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: a.py

import sys
from import_profiler import profile_import
import tensorflow
with profile_import() as context:
    import tensorflow.contrib
context.print_info(threshold=5)
sys.exit()



import tensorflow.contrib.framework
import cv2
with profile_import() as context:
    import tensorpack
context.print_info(threshold=5)
sys.exit()



import tensorflow.contrib.framework
import cv2
import time
s = time.time()
import tensorpack
print(time.time() - s)

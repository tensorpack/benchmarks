#!/usr/bin/env python
# flake8: noqa
# -*- coding: utf-8 -*-
# File: profile-import.py

import sys
from import_profiler import profile_import

if __name__ == '__main__':
    task = sys.argv[1]

    if task == 'contrib':
        import tensorflow
        with profile_import() as context:
            import tensorflow.contrib
        context.print_info(threshold=5)
    elif task == 'tensorpack':
        import tensorflow.contrib.framework
        import cv2
        with profile_import() as context:
            import tensorpack
        context.print_info(threshold=5)
    elif task == 'timing':
        import tensorflow.contrib.framework
        import cv2
        import time
        s = time.time()
        import tensorpack
        print(time.time() - s)

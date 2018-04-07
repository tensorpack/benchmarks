

Some scripts to test ImageNet reading speed.

Best speed on ImageNet:

+ With `tensorpack.dataflow` (pure python loader):

Augmentations=[GoogleNetResize, Lighting, Flip]: reach 5.6k image/s on a DGX1.

+ With `tf.data`:

Augmentations=[GoogleNetResize, Flip]: 10k image/s on a DGX1.

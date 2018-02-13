
## DCGAN
Environment: TFv1.3.0-rc1-1302-g593dc8e. Tesla P100.

Code: [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow/) at commit b13830.

* DCGAN-tensorflow:
```
python main.py --dataset celebA --train --crop
```
This command takes 0.36s per iteration, where each iteration is 1 update to D and 2 updates to G.

* [tensorpack DCGAN examples](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/GAN/DCGAN.py):

Modify the code to use `SeparateGANTrainer(..., d_period=2)`, and run with:
```
python DCGAN.py --data /path/to/img_align_celebA --crop-size 108 --batch 64
```

This script runs at 15.5 it/s, where every two iterations is equivalent to 1 iteration in DCGAN-tensorflow.
Therefore this script is roughly 2.8x faster.

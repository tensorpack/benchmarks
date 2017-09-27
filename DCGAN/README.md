
## DCGAN

Environment: TF1.3. Tesla P100.
Code: [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow/) at commit b13830.

* DCGAN-tensorflow:
```
python main.py --dataset celebA --train --crop
```
This command takes 0.36s per iteration, where each iteration is 1 update to D and 2 updates to G.

* [tensorpack DCGAN examples](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/GAN/DCGAN.py):

Apply the following diff to make a fair comparison:
```diff
18c18
< from GAN import SeparateGANTrainer, RandomZData, GANModelDesc
---
> from GAN import GANTrainer, RandomZData, GANModelDesc
39c39
< opt.BATCH = 64
---
> opt.BATCH = 128
166c166
<         SeparateGANTrainer(config, d_period=2).train()
---
>         GANTrainer(config).train()
```

And run with:
```
python DCGAN.py --data /path/to/img_align_celebA --crop-size 108
```

This script runs at 15.5 it/s, where every two iterations is equivalent to 1 iteration in DCGAN-tensorflow.
Therefore this script is roughly 2.8x faster.

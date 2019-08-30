## Mask R-CNN

### Environment:

* TensorFlow 1.14 (6e0893c79) + PR30893
* Python 3.7
* CUDA 10.0, CuDNN 7.6.2
* tensorpack 0.9.7.1 (a7f4094d)
* keras 2.2.5
* matterport/Mask_RCNN 3deaec5d
* horovod 0.18.0
* 8xV100s + 80xE5-2698 v4

### Settings:
* Use the standard hyperparameters used by [Detectron](https://github.com/facebookresearch/Detectron/),
  except that total batch size is set to 8.

* `export TF_CUDNN_USE_AUTOTUNE=0` to avoid CuDNN warmup time.

* Measure speed using "images per second", in the second or later epochs.


### [tensorpack FasterRCNN example](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN):

Using `TRAINER=replicated`, the speed is about 42 img/s:
```
./train.py  --config DATA.BASEDIR=~/data/coco DATA.NUM_WORKERS=20 MODE_FPN=True --load ImageNet-R50-AlignPadding.npz
```

Using `TRAINER=horovod`, the speed is about 50 img/s:
```
mpirun -np 8 ./train.py --config DATA.BASEDIR=~/data/coco MODE_FPN=True TRAINER=horovod --load ImageNet-R50-AlignPadding.npz
```

### [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/):

Apply [maskrcnn.patch](maskrcnn.patch) to make it use the same hyperparameters.
Then, run command:

```
python coco.py train --dataset=~/data/coco/ --model=imagenet
```

It trains at 0.77 ms / steps, aka 10 img/s.


### Note:

* Mask R-CNN is a complicated system and there could be many implementation differences.
  The above diff only makes the two systems perform roughly the same training.

* The training time of a R-CNN typically slowly decreases as the training progresses.
  In this experiment we only look at the training time of the first couple thousand iterations.
  It cannot be extrapolated to compute the total training time of the model.

* Tensorpack's Mask R-CNN is not only fast, but also
  [more accurate](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN#results).

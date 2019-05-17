## Benchmark CNN with other TF high-level APIs

Tensorpack is __1.2x~5x__ faster than the equivalent code written in some other TF high-level APIs.

### Benchmark setting:

* Hardware: AWS p3.16xlarge (8 Tesla V100s)
* Software:
Python 3.6, TF 1.13.1, cuda 10, cudnn 7.4.2, Keras 2.1.5, tflearn 0.3.2, tensorpack 0.9.4.
* Measurement: speed is measured by images per second (__larger is better__). First epoch is warmup and
	is not considered in timing. Second or later epochs have statistically insignificant difference.
* Data:
	* True data for Cifar10.
	* For ImageNet, assumed to be a constant numpy array already available on CPU.
		This is a reasonable setting because data always has to come from somewhere to CPU anyway.
* __Source code is here__. They are all <100 lines that you can easily
  run and __verify by yourself__.

### On a Single GPU:
| Task                           | tensorpack  | Keras  | tflearn |
| ------------------------------ | ----------- | ------ | ------- |
| Keras Official Cifar10 Example | __11904__   | 7142   | 5882    |
| VGG16 on fake ImageNet         | __230__     | 204    | 194     |
| AlexNet on fake ImageNet       | __2603__    | 1454   | N/A     |
| ResNet50 on fake ImageNet      | __333__     | 266    | N/A     |

### Data Parallel on 8 GPUs:

Each script used in this section can be run with "./script.py NUM_GPU" to use a different number of GPUs.

|                      | 1 GPU   | 2 GPUs | 8 GPUs    |
| -------------------- | ------- | ------ | --------- |
| tensorpack+ResNet50  | 333     | 579    | __2245__  |
| Keras+ResNet50       | 266     | 320    | 470       |
|                      |         |        |           |
| tensorpack+VGG16     | 230     | 438    | __1651__  |
| Keras+VGG16          | 204     | 304    | 449       |



Notes:

1. With a better (but different in batch sizes, etc) setting in [../ResNet-MultiGPU/](../ResNet-MultiGPU/),
	tensorpack can further reach 2800 im/s for ResNet50 on a p3.16xlarge instance.
	And 9225 im/s with all optimizations + fp16 turned on.
2. It's possible for Keras to be faster (by using better input pipeline, building data-parallel graph by yourself, etc), but it's NOT
	how most users are using Keras or how any of the Keras examples are written.

	Using Keras together with Tensorpack is one way to make Keras faster.
	See the [Keras+Tensorpack example](https://github.com/tensorpack/tensorpack/tree/master/examples/keras).

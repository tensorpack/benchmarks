## Benchmark CNN with other TF wrappers

### Benchmark setting:

* Hardware: AWS p3.16xlarge (8 Tesla V100s)
* Software:
Python 3.6, TF 1.6.0, cuda 9, cudnn 7.0.5, Keras 2.1.5, tflearn 0.3.2, tensorpack 0.8.3.
* Measurement: speed is measured by images per second. First epoch is warmup and
	is not considered in timing. Second or later epochs have statistically insignificant difference.
* Data:
	* True data for Cifar10.
	* For ImageNet, assumed to be a constant numpy array already available on CPU.
		This is a reasonable setting because data always has to come from somewhere to CPU anyway.
* All sources are here. They are all <100 lines of code that you can easily run and verify.

### On a Single GPU:
| Task											     | tensorpack	 | Keras	| tflearn  |
| ------------------------------ | ----------- | ------ | -------  |
| Keras Official Cifar10 Example |	__7507__   | 3448   | 3967     |
| VGG16 on fake ImageNet			   |	__226__		 | 188		| 114      |
| AlexNet on fake ImageNet	     |	__2633__	 | 1280		| N/A      |
| ResNet50 on fake ImageNet	     |	__318__	   | 230		| N/A      |

### Data Parallel on 8 GPUs:

Each script has one line to change the number of GPUs.

|						           | 1 GPU   | 2 GPUs  | 8 GPUs    |
| -------------------- | ------- | ------  | --------- |
| tensorpack+ResNet50  | 318     |	582	   | __2177__  |
| Keras+ResNet50		   | 230     |	291		 |  376      |
| |
| tensorpack+VGG16     | 226     |	438	   | __1471__  |
| Keras+VGG16			     | 188     |	320	   |   501     |



Notes:

1. With a better (but different in batch sizes, etc) setting in [../ResNet-MultiGPU/](../ResNet-MultiGPU/),
	tensorpack can further reach 2600 im/s for ResNet50 on a p3.16xlarge instance.
2. You can train a Keras model in tensorpack to make it faster.
See [Keras+Tensorpack example](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/keras).

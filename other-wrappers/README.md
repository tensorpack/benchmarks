## Benchmark CNN with other TF wrappers

Each subdirectory contains a group of test scripts,
written with simplicity so that people can easily run by themselves.

### Benchmark setting:

* Software: The following software versions were used to get claimed numbers:
TF 1.6.0, cudnn 7.0.5, Keras 2.1.5, tflearn 0.3.2, tensorpack commit 70d95c1.
* Measurement: speed is measured by images per second. First epoch is warmup and
	is not considered in timing.
* Data: True data for Cifar10. For ImageNet, assumed to be a constant ndarray already available on CPU.

### On a Single P100:
| Task											  | tensorpack	 | Keras	  | tflearn  |
| --------------------------- | ------------ | ------   | -------  |
| Small CNN on Cifar10 			  |		__7267__   | 3846     | 4084     |
| VGG16 on fake ImageNet			|		__146__		 | 133			| 114      |
| AlexNet on fake ImageNet	  |		__2067__	 | 1000			| N/A      |
| ResNet50 on fake ImageNet	  |		__213__	   | 167			| N/A      |

### ResNet50 Data Parallel on 8 P100s:

It takes one line of code change in each script to change the number of GPUs.

|						  | 1 GPU   | 2 GPU  | 8 GPU |
| ----------- | ------- | ------ | ------|
| tensorpack  | 213     |	385	   | 1333  |
| Keras			  | 167     |	218		 |  377  |

Notes:

1. Keras and tflearn scripts are copied from their official examples and slightly modified
	to make sure they are doing equivalent work.
2. With a better (but not equivalent) setting in [../ResNet-MultiGPU/](../ResNet-MultiGPU/),
	tensorpack can actually reach 1700 im/s for ResNet50.
3. You can train a Keras model in tensorpack and enjoy its speed.
See [Keras+Tensorpack example](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/keras).

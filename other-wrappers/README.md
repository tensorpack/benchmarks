## Benchmark CNN with other TF wrappers

### Benchmark setting:

* Software: The following software versions were used to get numbers below:
Python 3, TF 1.6.0, cudnn 7.0.5, Keras 2.1.5, tflearn 0.3.2, tensorpack commit 70d95c1.
* Measurement: speed is measured by images per second. First epoch is warmup and
	is not considered in timing.
* Data: True data for Cifar10. For ImageNet, assumed to be a constant numpy array already available on CPU.
* All sources are here. They are all <100 lines of code that you can easily run and verify.

### On a Single P100:
| Task											     | tensorpack	 | Keras	| tflearn  |
| ------------------------------ | ----------- | ------ | -------  |
| Keras Official Cifar10 Example |	__7575__   | 3389   | 4042     |
| VGG16 on fake ImageNet			   |	__146__		 | 133		| 114      |
| AlexNet on fake ImageNet	     |	__2067__	 | 1000		| N/A      |
| ResNet50 on fake ImageNet	     |	__213__	   | 167		| N/A      |

### ResNet50 Data Parallel on 8 P100s:

Each script has one line to change the number of GPUs.

|						  | 1 GPU   | 2 GPUs | 8 GPUs |
| ----------- | ------- | ------ | ------ |
| tensorpack  | 213     |	385	   | 1333   |
| Keras			  | 167     |	218		 |  377   |

Notes:

1. With a better (but not equivalent) setting in [../ResNet-MultiGPU/](../ResNet-MultiGPU/),
	tensorpack can actually reach 1600 im/s for ResNet50.
2. You can train a Keras model in tensorpack to make it faster.
See [Keras+Tensorpack example](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/keras).

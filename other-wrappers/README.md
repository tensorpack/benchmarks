## Benchmark CNN with other TF wrappers

Each subdirectory contains a group of test scripts.
I use the following software versions to get claimed numbers:

TF 1.6.0, cudnn 7.0.5, Keras 2.1.5, tflearn 0.3.2, tensorpack commit 70d95c1.

Tested on 03/08/2018 on a single P100.

Speed below is measured by images per second.

| Task											  | tensorpack	 | Keras	  | tflearn  |
| --------------------------- | ------------ | ------   | -------  |
| Small CNN on Cifar10 			  |		__7267__   | 3846     | 4084     |
| VGG16 on fake ImageNet			|		__146__		 | 133			| 114      |
| AlexNet on fake ImageNet	  |		__2067__	 | 1000			| N/A      |
| ResNet50 on fake ImageNet	  |		__220__	   | 167			| N/A      |

Note:

1. The first-epoch is warmup and is not considered in timing.
2. Data is assumed to be always ready on CPU.
3. Keras and tflearn scripts are copied from their official examples and slightly modified
	to make sure they are doing equivalent work.

I would hope to benchmark on real ImageNet, but I couldn't find any
working training code on ImageNet with Keras/tflearn.

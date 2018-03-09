## Benchmark CNN with other TF wrappers

Each subdirectory contains a group of test scripts.
Please use the following software versions to get claimed numbers: TF1.6.0, cudnn7.0.

Tested on 03/08/2018 on a single P100.

Speed below is measured by images per second.

| Task											  | tensorpack	 | Keras	  | tflearn  |
| --------------------------- | ------------ | ------   | -------  |
| Small CNN on Cifar10 			  |		__7267__   | 3846     | 4084     |
| VGG on fake ImageNet			  |		__146__		 | 133			| 114      |
| AlexNet on fake ImageNet	  |		__2067__	 | 1000			| didn't test|

Note:

1. The first-epoch is warmup and is not considered in timing.
2. Data is assumed to be a constant numpy array on CPU.
3. Keras and tflearn scripts are copied and slightly modified from their official examples
	to make sure they are doing equivalent work.
4. This tensorpack script uses NCHW format as suggested by TensorFlow.
	 The Keras script uses the best one after trying both. tflearn does not have such options.

I would hope to benchmark on real ImageNet, but sadly I couldn't find any
working training code on ImageNet with Keras/tflearn.

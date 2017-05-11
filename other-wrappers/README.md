
## Benchmark tensorpack with other TF wrappers

Each subdirectory contains a group of test scripts.

Environment: TF1.1, Single Tesla M40, cudnn5.1.

They are trained with same configurations.  Speed is measured by images per second.

| Task											  | tensorpack	 | Keras	  | tflearn  |
| --------------------------- | ------------ | ------   | -------  |
| Small CNN on Cifar10 			  |		6250	 | 4166 | 4166 |
| VGG on fake ImageNet			  |		74		 | 47   | 48   |
| AlexNet on fake ImageNet	  |		1280	 | 711  | didn't test|

Note that the first-epoch is taken as warmup and not considered in timing.

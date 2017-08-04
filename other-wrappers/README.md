
## Benchmark tensorpack with other TF wrappers

Each subdirectory contains a group of test scripts.

Environment: TF1.3.0rc1, Single Tesla M40, cudnn6.0.

They are trained with same configurations.  Speed is measured by images per second.

| Task											  | tensorpack	 | Keras	  | tflearn  |
| --------------------------- | ------------ | ------   | -------  |
| Small CNN on Cifar10 			  |		5885       | 3571     | 3571     |
| VGG on fake ImageNet			  |		74				 | 49				| 49       |
| AlexNet on fake ImageNet	  |		1212			 | 711			| didn't test|

Note that the first-epoch is taken as warmup and not considered in timing.

I would hope to benchmark on real ImageNet, but sadly I couldn't find any
working training code on ImageNet with Keras/tflearn.

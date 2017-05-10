
## Benchmark tensorpack with other TF wrappers

Each subdirectory contains a group of test scripts.

Environment: TF1.1, Tesla M40, cudnn5.1

| Task											  | tensorpack	 | Keras	| tflearn |
| --------------------------- | ------------ | ------ | ------- |
| Small CNN on Cifar10 			  |			8s/ep		 | 12s/ep | 12s/ep  |
| VGG on fake(small) ImageNet |		43s/ep		 | 56s/ep | 66s/ep  |


Note that the first-epoch is taken as warmup and not considered in timing.

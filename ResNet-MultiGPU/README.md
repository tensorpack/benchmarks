
# Benchmark MultiGPU training against tensorflow/benchmarks

tensorpack MultiGPU trainers are implemented following the awesome examples in
[tensorflow/benchmarks](github.com/tensorflow/benchmarks).
Their performance should be the same.

## Test environment:
* TF Version: v1.3.0-rc1-1302-g593dc8e
* Machine: Nvidia DGX1
Note that these experiements are run on machines of the same configurations,
but not the same set of physical machines. So expect some variance in results.

## Scripts:

The following commands in `tensorflow/benchmarks`
```
python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=64 --model=resnet50 --variable_update=replicated/parameter_server --local_parameter_device=cpu
```

is roughly the same as this tensorpack script:
```
python resnet-multigpu.py --fake-location gpu/cpu/python --variable_update=replicated/parameter_server
```

There are tiny differences in the way data is synthesized (the `--fake-location` option):
* gpu: synthesized on GPU as a constant. TF benchmark uses something similar to this but not the same.
* cpu: synthesized on CPU inside TF runtime.
* python: synthesized on CPU inside Python (and copied to TF), with gpu prefetch.
This is the recommended experiement setting in tensorpack, because it's easy to write
(use Python to write data loader) and still fast.

## Performance (image/second):

variable_update=replicated:

| #GPU			| tensorpack(GPU/CPU/Python) | tensorflow/benchmarks |
| --------- | ----------------------	| --------------------  |
| 1         |	228/228/219							| 225.73								|
| 2					|	427/423/415   				  | 424.54								|
| 4					| 802/785/787							|	789.51								|
| 8					|	1612/1579/1551					|	1580.58								|


variable_update=parameter_server:

| #GPU			| tensorpack(GPU/CPU/Python) | tensorflow/benchmarks  |
| --------- | -------------------				 | --------------------   |
| 1         |	227/227/223								 |  221.68								|
| 2					|	428/418/403								 |  421.01								|
| 4					|	817/802/787								 |	828.29								|
| 8					|	1651/1556/1574	  				 |	1604.55								|

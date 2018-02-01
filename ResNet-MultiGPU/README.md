
# Benchmark MultiGPU training against tensorflow/benchmarks

Tensorpack MultiGPU trainers are implemented following the awesome examples in
[tensorflow/benchmarks](https://github.com/tensorflow/benchmarks).
Their performance should be the same.

This script is focused on the performance of different parallel strategies.
To train on real data with reasonable experiments settings, see [ResNet-Horovod](../ResNet-Horovod).

## Scripts:

The following commands in `tensorflow/benchmarks`
```
python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=64 --model=resnet50 --variable_update=replicated/parameter_server --local_parameter_device=cpu
```

is roughly the same as this tensorpack script:
```
python resnet-multigpu.py --fake-location gpu --variable-update=replicated/parameter_server
```

There are tiny differences in the way data is synthesized (the `--fake-location` option):
* gpu: synthesized on GPU as a constant. TF benchmark uses something most similar to this.
* cpu: synthesized on CPU inside TF runtime.
* python-queue: synthesized on CPU inside Python, transferred to TF with queue, prefetch on GPU.
This is the most common experiement setting in tensorpack, because it's easy to write
(use Python to write data loader) and still fast.
* python-dataset: also use python to write data loader, but transfer to TF with `tf.data + prefetch` instead.
* zmq-consume: consume data from a ZMQ pipe, using [my zmq ops](https://github.com/tensorpack/zmq_ops), also with GPU prefetch.
	The data producer can therefore be written in any language.

Data processing inside TF is usually a [bad idea](http://tensorpack.readthedocs.io/en/latest/tutorial/input-source.html#python-reader-or-tf-reader).
When data comes from outside TF, my experiements show
that `zmq-consume` is the fastest input pipeline compared to others here.
It's also faster than `tensorflow/benchmarks` (tested on Jan 6 2018 with TF1.5.0rc0) when training real data.

## Performance (image/second):

The following was tested with: TF v1.3.0-rc1-1302-g593dc8e on a single DGX1.
Experiments were not run for multiple times. So expect some variance in results.

variable-update=replicated:

| #GPU			| tensorpack(GPU/CPU/Python) | tensorflow/benchmarks |
| --------- | ----------------------	| --------------------  |
| 1         |	228/228/219							| 225.73								|
| 2					|	427/423/415   				  | 424.54								|
| 4					| 802/785/787							|	789.51								|
| 8					|	1612/1579/1551					|	1580.58								|

variable-update=parameter_server:

| #GPU			| tensorpack(GPU/CPU/Python) | tensorflow/benchmarks  |
| --------- | -------------------				 | --------------------   |
| 1         |	227/227/223								 |  221.68								|
| 2					|	428/418/403								 |  421.01								|
| 4					|	817/802/787								 |	828.29								|
| 8					|	1651/1556/1574	  				 |	1604.55								|

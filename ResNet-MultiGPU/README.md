
# Benchmark Multi-GPU training against tensorflow/benchmarks

Tensorpack Multi-GPU trainers are implemented following the awesome examples in
[tensorflow/benchmarks](https://github.com/tensorflow/benchmarks).
Their performance should be the same.

Here we measure performance by the number of images the trainer can process per second when training a ResNet-50 on ImageNet-size images.

This script is tested on fake data to focus on the performance of different parallel strategies.
To train on real data with reasonable experiment settings, see
[Tensorpack ResNet example](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet) or [ResNet-Horovod benchmark](../ResNet-Horovod).

## Scripts:

The following command in tensorflow/benchmarks:
```
python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=64 --model=resnet50 --variable_update=replicated/parameter_server --local_parameter_device=cpu
```

is roughly the same as this command in tensorpack:
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

Data processing inside TF is often a [bad idea in practice](https://tensorpack.readthedocs.io/tutorial/philosophy/dataflow.html#alternative-data-loading-solutions).
When data comes from outside TF, my experiements show
that `zmq-consume` is the fastest input pipeline compared to others here.
It's also faster than `tensorflow/benchmarks` (tested on Jan 6 2018 with TF1.5.0rc0) when training real data.

## Performance @ Sep 2017:

Environment:
* Software: TF v1.3.0-rc1-1302-g593dc8e; tensorpack 0.5.
* Hardware: 8 P100s.

Note that the latest source code uses new features in tensorpack and therefore won't work with tensorpack 0.5.
Checkout an old version if you intend to repdouce these numbers.

Experiments were not run for multiple times. So expect some small variance in results.

`variable-update=replicated`:

| #GPU      | tensorpack(GPU/CPU/Python) | tensorflow/benchmarks |
| --------- | ----------------------     | --------------------  |
| 1         | 228/228/219                | 225.73                |
| 2         | 427/423/415                | 424.54                |
| 4         | 802/785/787                | 789.51                |
| 8         | 1612/1579/1551             | 1580.58               |

`variable-update=parameter_server`:

| #GPU      | tensorpack(GPU/CPU/Python) | tensorflow/benchmarks |
| --------- | -------------------        | --------------------  |
| 1         | 227/227/223                | 221.68                |
| 2         | 428/418/403                | 421.01                |
| 4         | 817/802/787                | 828.29                |
| 8         | 1651/1556/1574             | 1604.55               |

## Performance @ May 2019:

Environment:

* Software: TF 1.13.1, cuda 10, cudnn 7.4.2, tensorpack 0.9.5.
* Hardware: AWS p2.16xlarge (8 V100s)

Results:

* `--fake-location=gpu --variable-update=horovod`: 2874 img/s.
* `--fake-location=gpu --variable-update=replicated`: 2844 img/s.
* `--fake-location=gpu --variable-update=replicated --use-fp16`: 5224 img/s.
* `--fake-location=gpu --variable-update=replicated --use-fp16 --batch 128`: 5891 img/s
* `--fake-location=gpu --variable-update=replicated --use-fp16 --batch 128 --use-xla-compile`: 9225 img/s

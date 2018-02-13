
# Tensorpack + Horovod

Multi-GPU / distributed training on ImageNet.

It follows most settings in the paper
+ [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

## Usage:

```bash
# Single Machine:
$ ./serve-data.py --data ~/data/imagenet/ --batch 64
$ mpirun -np 8 --output-filename test.log python3 ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64
```

```bash
# Multiple machines with RoCE/IB:
host1$ ./serve-data.py --data ~/data/imagenet/ --batch 64
host2$ ./serve-data.py --data ~/data/imagenet/ --batch 64
mpirun -np 16 -H host1:8,host2:8 --output-filename test.log \
		-bind-to none -map-by slot \
		-mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
		-x PATH -x PYTHONPATH -x LD_LIBRARY_PATH python3 -x NCCL_DEBUG=INFO \
		./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64
```

Notes:
1. MPI does not like fork(), so running `serve-data.py` inside MPI is not a good idea.
2. Remove some MPI arguments if running with plain TCP.
   See https://github.com/uber/horovod/blob/master/docs/benchmarks.md for details.
	 Performance will be bad.

```bash
# Benchmark data speed:
$ ./serve-data.py --data ~/data/imagenet/ --batch 64 --benchmark
# Benchmark training with fake data: train with `--fake`.
```

## Distributed ResNet50 Results:

Validation time excluded from total time.

|Devices    | per GPU batch | time    | top1 acc |
|   -       |    -          |    -										|   -      |
| 128 P100s |	32					  | 1h40min  								|  23.62%   |
| 128 P100s |	64					  | 1h23min  								|  23.97%   |
| 256 P100s |	32					  | 1h9min  								|  23.90%   |


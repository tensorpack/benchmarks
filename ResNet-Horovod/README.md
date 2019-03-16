
# Tensorpack + Horovod

Multi-GPU / distributed training on ImageNet, with TensorFlow + Tensorpack + Horovod.

It reproduces the settings in the paper
+ [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

The code is annotated with sentences from the paper.

Based on this baseline implementation, we implemented adversarial training and obtained ImageNet classifiers with state-of-the-art adversarial robustness. See our code release at [facebookresearch/ImageNet-Adversarial-Training](https://github.com/facebookresearch/ImageNet-Adversarial-Training/). 

## Dependencies:
+ TensorFlow>=1.5, tensorpack>=0.8.5.
+ [Horovod](https://github.com/uber/horovod) with NCCL support.
	See [doc](https://github.com/uber/horovod/blob/master/docs/gpus.md) for its installation instructions.
+ [zmq_ops](https://github.com/tensorpack/zmq_ops): optional but recommended.
+ Prepare ImageNet data into [this structure](http://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12).

## Run:
```bash
# Single Machine, Multiple GPUs:
# Run the following two commands together:
$ ./serve-data.py --data ~/data/imagenet/ --batch 64
$ mpirun -np 8 --output-filename test.log python3 ./imagenet-resnet-horovod.py -d 50 --data ~/data/imagenet/ --batch 64
```

```bash
# Multiple Machines with RoCE/IB:
host1$ ./serve-data.py --data ~/data/imagenet/ --batch 64
host2$ ./serve-data.py --data ~/data/imagenet/ --batch 64
$ mpirun -np 16 -H host1:8,host2:8 --output-filename test.log \
		-bind-to none -map-by slot -mca pml ob1 \
	  -x NCCL_IB_CUDA_SUPPORT=1 -x NCCL_IB_DISABLE=0 -x NCCL_DEBUG=INFO \
		-x PATH -x PYTHONPATH -x LD_LIBRARY_PATH \
		python3 ./imagenet-resnet-horovod.py -d 50 \
        --data ~/data/imagenet/ --batch 64 --validation distributed
```

Notes:
1. MPI does not like fork(), so running `serve-data.py` inside MPI is not a good idea.
2. You may tune the best mca & NCCL options for your own systems.
   See [horovod docs](https://github.com/uber/horovod/blob/master/docs/) for details.
   Note that TCP connection will then have much worse scaling efficiency.
3. To train on small datasets, __you don't need a separate data serving process or zmq ops__.
	You can simply load data inside each training process with its own data loader.
	The main motivation to use a separate data loader is to avoid fork() inside
	MPI and to make it easier to benchmark.
4. You can pass `--no-zmq-ops` to both scripts, to use Python for communication instead of the faster zmq_ops.
5. If you're using slurm in a cluster, checkout an example [sbatch script](slurm.script).

## Performance Benchmark:
```bash
# To benchmark data speed:
$ ./serve-data.py --data ~/data/imagenet/ --batch 64 --benchmark
# To benchmark training with fake data:
# Run the training command with `--fake`
```

## Distributed ResNet50 Results:

 | devices   | batch per GPU | time   <sup>[1](#ft1)</sup> | top1 err <sup>[3](#ft3)</sup>|
 | -         | -             | -                           | -        |
 | 32 P100s  | 64            | 5h9min                      | 23.73%   |
 | 128 P100s | 32            | 1h40min                     | 23.62%   |
 | 128 P100s | 64            | 1h23min                     | 23.97%   |
 | 256 P100s | 32            | 1h9min <sup>[2](#ft2)</sup> | 23.90%   |


<a id="ft1">1</a>: Validation time excluded from total time. Time depends on your hardware.

<a id="ft2">2</a>: This corresponds to exactly the "1 hour" setting in the original paper.

<a id="ft3">3</a>: The final error typically has ±0.1 or more fluctuation according to the paper.

Although the code does not scale very ideally with 32 machines, it does scale with 90+% efficiency on 2 or 4 machines.

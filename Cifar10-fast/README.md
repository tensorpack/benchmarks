
# Train to 94% accuracy on Cifar10 within a minute

This script is able to train to 94% accuracy (average over multiple runs)
on Cifar10 within a minute, when run with:

* 1 V100 GPU
* TensorFlow 1.14
* Tensorpack @047579df
* CUDA 10, CuDNN 7.6.2

The script mostly follows the [cifar10-fast repo](https://github.com/davidcpage/cifar10-fast)
with small modifications on architecture.

This sort of "competition" doesn't really constitute any innovations since it's
mainly about recipe tuning and overfitting the test accuracy.
But since someone has tuned it, it's an interesting excercise to follow.

## To Run:
```
./cifar10-fast.py --num-runs 10
```

Time: it takes about 2.2s per epoch over the 24-epoch training.
The first epoch is slower because of CuDNN warmup and XLA compilation.

Accuracy: it prints the test accuracy after every run finishes, and the average accuracy in the end.

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import time
from contextlib import contextmanager
from tensorpack import *

@contextmanager
def benchmark(name="unnamed context"):
    elapsed = time.time()
    yield
    elapsed = time.time() - elapsed
    print('[{}] finished in {} ms'.format(name, int(elapsed * 1000)))

SHAPE = [64, 225, 225, 3]
def f(tensor):
    tensor = tf.reshape(tensor, SHAPE)
    tensor = Conv2D('conv', tensor, 64, 3)
    tensor = Conv2D('conv2', tensor, 64, 3)
    tensor = Conv2D('conv3', tensor, 64, 3)
    return tf.reduce_sum(tensor)

NR_RUN = 40



def benchmark_FIFO():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            dummy = tf.random_normal(SHAPE, dtype=tf.float32, stddev=1e-1)
            stager = tf.FIFOQueue(NR_RUN + 500, [tf.float32])
            enqeue_op = stager.enqueue([dummy])
            deq = stager.dequeue()

        with tf.device('/gpu:0'):
            dequeue_op = f(deq)
            dequeue_op = tf.group(*[dequeue_op])

        sess.run(tf.global_variables_initializer())
        print("Queue Enqueue ...")
        for i in range(NR_RUN + 20):
            sess.run(enqeue_op)

        print("Queue Run ...")
        for i in range(10):
            sess.run(dequeue_op)
        with benchmark("FIFO"):
            for i in range(NR_RUN):
                sess.run(dequeue_op)


def benchmark_staging():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            dummy = tf.random_normal(SHAPE, dtype=tf.float32, stddev=1e-1)
        with tf.device('/gpu:0'):
            stager = data_flow_ops.StagingArea([tf.float32])
            enqueue_op = stager.put([dummy])
            dequeue_op = f(stager.get())
            dequeue_op = tf.group(*[dequeue_op, enqueue_op])

        sess.run(tf.global_variables_initializer())
        print("Staging Enqueue ...")
        for i in range(20):
            sess.run(enqueue_op)

        print("Staging Run ...")
        for i in range(10):
            sess.run(dequeue_op)
        with benchmark("staging"):
            for i in range(NR_RUN):
                sess.run(dequeue_op)


def benchmark_FIFOstaging():
    tf.reset_default_graph()

    with tf.device('/cpu:0'):
        dummy = tf.random_normal(SHAPE, dtype=tf.float32, stddev=1e-1)
        stager = tf.FIFOQueue(NR_RUN + 500, [tf.float32])
        enqueue_op1 = stager.enqueue([dummy])
        dequeue = stager.dequeue()

    with tf.device('/cpu:0'):
        stager = data_flow_ops.StagingArea([tf.float32])
        enqueue_op2 = stager.put([dequeue])
        dequeue_op2 = stager.get()

    with tf.device('/gpu:0'):
        stager = data_flow_ops.StagingArea([tf.float32])
        enqueue_op = stager.put([dequeue_op2])
        dequeue_op = f(stager.get())
        dequeue_op = tf.group(*[dequeue_op, enqueue_op])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("FIFOStaging Enqueue ...")
        for i in range(NR_RUN + 40):
            sess.run(enqueue_op1)

        for i in range(NR_RUN + 35):
            sess.run(enqueue_op2)
        for i in range(20):
            sess.run(enqueue_op)

        print("FIFOStaging Run ...")
        for i in range(10):
            sess.run(dequeue_op)
        with benchmark("FIFOstaging"):
            for i in range(NR_RUN):
                sess.run(dequeue_op)

benchmark_FIFOstaging()
benchmark_staging()
benchmark_FIFO()

#!/usr/bin/env python

import keras
keras.backend.set_image_data_format('channels_first')
from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils
import numpy as np
import sys

try:
    NUM_GPU = int(sys.argv[1])
except IndexError:
    NUM_GPU = 1

batch_size = 32 * NUM_GPU

img_rows, img_cols = 224, 224

if keras.backend.image_data_format() == 'channels_first':
    X_train = np.random.random((batch_size, 3, img_rows, img_cols)).astype('float32')
else:
    X_train = np.random.random((batch_size, img_rows, img_cols, 3)).astype('float32')
Y_train = np.random.random((batch_size,)).astype('int32')
Y_train = np_utils.to_categorical(Y_train, 1000)


def gen():
    while True:
        yield (X_train, Y_train)


model = ResNet50(weights=None, input_shape=X_train.shape[1:])

if NUM_GPU != 1:
    model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)

for l in model.layers:
    print(l.input_shape, l.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

model.fit_generator(gen(), epochs=100, steps_per_epoch=50)

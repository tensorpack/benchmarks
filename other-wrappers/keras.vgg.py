#!/usr/bin/env python

import keras
keras.backend.set_image_data_format('channels_first')
from keras.models import Model
from keras.utils import np_utils
from keras.layers import *
import numpy as np

NUM_GPU = 1
batch_size = 64 * NUM_GPU
nb_classes = 1000
nb_epoch = 200

img_rows, img_cols = 224, 224

if keras.backend.image_data_format() == 'channels_first':
    X_train = np.random.random((batch_size, 3, img_rows, img_cols))
else:
    X_train = np.random.random((batch_size, img_rows, img_cols, 3))
Y_train = np.random.random((batch_size,)).astype('int32')
Y_train = np_utils.to_categorical(Y_train, nb_classes)


def gen():
    while True:
        yield (X_train, Y_train)


img_input = Input(shape=X_train.shape[1:])
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(nb_classes, activation='softmax', name='predictions')(x)

model = Model(img_input, x, name='vgg16')

if NUM_GPU != 1:
    model = keras.utils.multi_gpu_model(model, NUM_GPU)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(gen(), epochs=nb_epoch, steps_per_epoch=50)

#!/usr/bin/env python

import keras
keras.backend.set_image_data_format('channels_first')
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
import numpy as np

batch_size = 64
nb_classes = 1000
nb_epoch = 200

img_rows, img_cols = 224, 224

if keras.backend.image_data_format() == 'channels_first':
    X_train = np.random.random((batch_size, 3, img_rows, img_cols)).astype('float32')
else:
    X_train = np.random.random((batch_size, img_rows, img_cols, 3)).astype('float32')
Y_train = np.random.random((batch_size,)).astype('int32')
Y_train = np_utils.to_categorical(Y_train, nb_classes)

def gen():
    while True:
        yield (X_train, Y_train)

model = Sequential()
model.add(Convolution2D(64, 11, strides=4, padding='valid', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Convolution2D(192, 5, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Convolution2D(384, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

for l in model.layers:
    print(l.input_shape, l.output_shape)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(gen(), epochs=nb_epoch, steps_per_epoch=200)

# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer


from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers import LSTM

from keras_extension.layers import MAC

# input_c = Input(shape=(10,))
# input_q = Input(shape=(20,))
# input_s = Input(shape=(7, 10))
# x = ControlUnit()([input_c, input_q, input_s])
# x = Dense(10)(x)
# output_layer = Activation('softmax')(x)
# model = Model([input_c, input_q, input_s], output_layer)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# print(model.summary())
#
#
# input_c = Input(shape=(10,))
# input_m = Input(shape=(10,))
# input_k = Input(shape=(7, 10))
# x = ReadUnit()([input_c, input_m, input_k])
# x = Dense(10)(x)
# output_layer = Activation('softmax')(x)
# model = Model([input_c, input_m, input_k], output_layer)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
#
# input_c = Input(shape=(10,))
# input_r = Input(shape=(10,))
# input_m = Input(shape=(10,))
# input_msa = Input(shape=(10,))
# x = WriteUnit()([input_c, input_r, input_m, input_msa])
# x = Dense(10)(x)
# output_layer = Activation('softmax')(x)
# model = Model([input_c, input_r, input_m, input_msa], output_layer)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
#
# from keras.layers import Add, Flatten
# input_c = Input(shape=(10,))
# input_m = Input(shape=(10,))
# input_q = Input(shape=(20,))
# input_cw = Input(shape=(15, 10,))
# input_k = Input(shape=(35, 10,))
# input_msa = Input(shape=(10,))
# x, y = MACCell()([input_c, input_m, input_q, input_cw, input_k, input_msa])
# z = Add()([x, y])
# z = Dense(10)(z)
# output_layer = Activation('softmax')(z)
# print(x, y, z, output_layer)
# model = Model([input_c, input_m, input_q, input_cw,
#                input_k, input_msa], output_layer)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()


from keras.layers import Add, Flatten

from keras.activations import relu
from keras_extension.activations import gelu

input_q = Input(shape=(20,))
input_cw = Input(shape=(15, 10,))
input_k = Input(shape=(35, 10,))
x, y = MAC(recurrent_length=3)([input_q, input_cw, input_k])
z = Add()([x, y])
z = Dense(10, activation=gelu)(z)
z = Activation(gelu)(z)
output_layer = Activation('softmax')(z)
model = Model([input_q, input_cw, input_k], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

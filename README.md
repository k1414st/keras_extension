Keras extension packages
========================

This package provide tips of keras extentions.

#### Installation
1. check out this repository.
```
git clone https://github.com/k1414st/keras_extension
```
2. use setup.py to install this package.
```
python setup.py install
```

PartialConvND (N = 1~3)
-----------------------
#### Description
PartialConvND classes are implementation of  "Partial Convolutional based padding" which re-weight convolution near image borders based on the ratios between the padded area and the convolution sliding window area. this algorithm has been introduced by *Guilin Liu et al.* in https://arxiv.org/abs/1811.11718

#### sample code
You can easily use PartialConv2D instead of Conv2D like this simple code.
```
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras_extension.layers import PartialConv2D

# load data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3) / 255.0
x_test = np.expand_dims(x_test, axis=3) / 255.0

# define model & compile
model = Sequential([
    # Conv2D(filters=3, kernel_size=(3, 3)),
    PartialConv2D(filters=3, kernel_size=(3, 3)),
    Dropout(0.2),
    # Conv2D(filters=3, kernel_size=(3, 3)),
    PartialConv2D(filters=3, kernel_size=(3, 3)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# run model (fit & evaluate)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
print(model.evaluate(x_test, y_test))
```

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Add
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model

from keras.layers import Conv2D
from keras.activations import relu
from keras_extension.layers import PartialConv2D
from keras_extension.activations import gelu

# constants of data shape.
# Width of picture(= Height), dimension of input data.
W = 28
D_input = 1


def _get_model(conv, activation, n_filters_base=8):
    """
    get model wrapper using convolution & activation layer.
    !!! convolution argument must be function of class constructor.

    This model exclude BN & Dropout layer for performance comparison.
    """
    input_layer = Input(shape=(W, W, D_input))
    x = input_layer
    n_filters_base = 16
    for i in range(3):
        n_filters = n_filters_base * (2 ** i)
        x = conv(n_filters)(x)
        # x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Dropout(0.2)(x)

    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    print(output_layer.shape)

    mdl = Model(input_layer, output_layer)
    mdl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    mdl.summary()
    return mdl


def _get_model_conv_relu():
    """ get model using Conv2D & RELU. """
    def conv(n_filters):
        return Conv2D(n_filters, kernel_size=(3, 3), padding='same', activation=None)
    activation = relu
    return _get_model(conv, activation)


def _get_model_partconv_gelu():
    """ get model using PartialConv2D & GELU. """
    def conv(n_filters):
        return PartialConv2D(n_filters, kernel_size=(3, 3), activation=None)
    activation = gelu
    return _get_model(conv, activation)


def _get_sample_data(random_state=None):
    """
    get mnist data
    input data is returned after normalization.
    objective data is returned after one-hot encoded.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=3) / 255.
    X_test = np.expand_dims(X_test, axis=3) / 255.
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return X_train, X_test, y_train, y_test


def _test_model(get_model_function):
    """ simple (mnist based) performance test. """

    # get simulated data, and split to train & test.
    X_train, X_test, y_train, y_test = _get_sample_data()

    # model fit & predict
    mdl = get_model_function()

    hist = mdl.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   batch_size=128, epochs=1, verbose=1)
    assert(hist.history['val_acc'][-1] > 0.95)


def test_conv_relu():
    _test_model(_get_model_conv_relu)


def test_partconv_gelu():
    _test_model(_get_model_partconv_gelu)


if __name__ == '__main__':
    # there seems to be a significant difference on "val_loss".
    test_conv_relu()  # compared test
    test_partconv_gelu()  # main test

import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input, Dense
from keras.models import Model
from keras_extension.layers import Lambda
# from keras.layers import Lambda

import tensorflow as tf


def get_model(test_list_case=False):
    """ get sparse-adaptive model """
    input_layer = Input(batch_shape=(2, 2048), sparse=True)
    x = input_layer
    # output_layer = Dense(units=1)(x)

    def sparse_matmul(x):
        w = tf.Variable(tf.zeros([2048, 1]), dtype=tf.float32)
        xx = tf.sparse.matmul(x, w)
        return xx

    output_layer = Lambda(sparse_matmul)(x)
    model = Model(input_layer, output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model


x = np.random.binomial(n=1, p=0.01, size=(100, 2048)).astype(np.float32)
y = np.random.binomial(n=1, p=0.5, size=(100, 1)).astype(np.float32)

def test_single_case():
    model = get_model()
    model.fit(csr_matrix(x), y, batch_size=2)
    return True

def test_list_case():
    model = get_model(test_list_case=True)
    model.fit(csr_matrix(x), y, batch_size=2)
    return True


if __name__ == '__main__':
    test_single_case()
    test_list_case()

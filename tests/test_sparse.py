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
        if not test_list_case:
            return tf.sparse.matmul(x, w)
        else:
            return tf.sparse.matmul(x[0], w)

    if not test_list_case:
        output_layer = Lambda(sparse_matmul)(x)
    else:
        output_layer = Lambda(sparse_matmul)([x, x])

    model = Model(input_layer, output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

# making dummy data (x is sparse matrix).
x = np.random.binomial(n=1, p=0.01, size=(100, 2048)).astype(np.float32)
x = csr_matrix(x)
y = np.random.binomial(n=1, p=0.5, size=(100, 1)).astype(np.float32)


def test_single_case():
    """ test of single_sparse_input_function model """
    model = get_model()
    model.fit(x, y, batch_size=2)
    return True

def test_list_case():
    """ test of sparse_list_input_function model """
    model = get_model(test_list_case=True)
    model.fit(x, y, batch_size=2)
    return True


if __name__ == '__main__':
    test_single_case()
    test_list_case()

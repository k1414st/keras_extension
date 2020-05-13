import numpy as np
from scipy.sparse.csr import csr_matrix
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_extension.layers import Lambda

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

SPARSITY = 0.01
# N_DATA = 100
# BATCH_SIZE = 2
N_DATA = 20
BATCH_SIZE = 20
DIM_SPARSE = 2**16

##################################################################################

def get_model(test_list_case=False):
    """ get sparse-adaptive model """
    input_layer = Input(batch_shape=(BATCH_SIZE, DIM_SPARSE), sparse=True)
    x = input_layer

    def sparse_matmul(x):
        # w = tf.Variable(tf.ones([DIM_SPARSE, 1]), dtype=tf.float32)
        w = tf.ones([DIM_SPARSE, 1], dtype=tf.float32)
        if not test_list_case:
            return tf.sparse.sparse_dense_matmul(x, w)
        else:
            return tf.sparse.sparse_dense_matmul(x[0], w)

    if not test_list_case:
        output_layer = Lambda(sparse_matmul)(x)
    else:
        output_layer = Lambda(sparse_matmul)([x, x])

    model = Model(input_layer, output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

# making dummy data (x is sparse matrix).
x = np.random.binomial(n=1, p=SPARSITY,
                       size=(N_DATA, DIM_SPARSE)).astype(np.float32)
x = csr_matrix(x)
y = np.random.binomial(n=1, p=0.5, size=(N_DATA, 1)).astype(np.float32)


def test_single_case():
    """ test of single_sparse_input_function model """
    model = get_model()
    model.fit(x, y, batch_size=BATCH_SIZE, steps_per_epoch=N_DATA)
    return True

def test_list_case():
    """ test of sparse_list_input_function model """
    model = get_model(test_list_case=True)
    model.fit(x, y, batch_size=BATCH_SIZE, steps_per_epoch=N_DATA)
    return True


if __name__ == '__main__':
    test_single_case()
    test_list_case()


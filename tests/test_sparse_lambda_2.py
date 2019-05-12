import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input, Dense
from keras.models import Model
from keras_extension.layers import Lambda

import keras.backend as K
import tensorflow as tf

from keras.engine.base_layer import InputSpec

SPARSITY = 0.01
N_DATA = 100
BATCH_SIZE = 2
DIM_SPARSE = 2**16

##################################################################################

def get_model():
    """ get sparse-adaptive model """
    input_layer = Input(batch_shape=(BATCH_SIZE, DIM_SPARSE), sparse=True)
    x = input_layer

    def sparse_matmul(x):
        print('X'*100)
        x = tf.sparse.reshape(x, [BATCH_SIZE, DIM_SPARSE])
        print('Y'*100)
        w = tf.Variable(tf.zeros([DIM_SPARSE, 2]), dtype=tf.float32)
        x = tf.sparse.matmul(x, w)
        return x

    # output_layer = Lambda(sparse_matmul)(x)
    x = Lambda(sparse_matmul)(x)

    x = Dense(1)(x)
    
    output_layer = x

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
    model.fit(x, y, batch_size=2)
    return True


if __name__ == '__main__':
    test_single_case()

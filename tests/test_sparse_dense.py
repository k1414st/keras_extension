import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input
from keras.models import Model
import tensorflow as tf

from keras_extension.layers import Dense

SPARSITY = 0.01
N_DATA = 100
BATCH_SIZE = 2
DIM_SPARSE = 2**16
N_EPOCH = 10

##################################################################################

def get_model(sparse=True):
    """ get sparse-adaptive model """
    input_layer = Input(batch_shape=(BATCH_SIZE, DIM_SPARSE), sparse=sparse)
    x = input_layer
    output_layer = Dense(units=1, use_bias=False,
                         activation=None, sparse=sparse)(x)

    model = Model(input_layer, output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

# making dummy data (x is sparse matrix).
x = np.random.binomial(n=1, p=SPARSITY,
                       size=(N_DATA, DIM_SPARSE)).astype(np.float32)
x_sp = csr_matrix(x)
y = np.random.binomial(n=1, p=0.5, size=(N_DATA, 1)).astype(np.float32)


def test_dense_sparse():
    """ test of sparsed dense model """
    model = get_model(sparse=True)
    model.fit(x_sp, y, batch_size=BATCH_SIZE, epochs=N_EPOCH)
    return True

def test_dense_nosparse():
    """ test of not-sparsed dense model """
    model = get_model(sparse=False)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=N_EPOCH)
    return True


if __name__ == '__main__':
    test_dense_sparse()
    test_dense_nosparse()

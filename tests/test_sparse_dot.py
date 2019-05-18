import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input, Flatten, Embedding, Dense, Lambda
from keras.models import Model
import tensorflow as tf

from keras_extension.layers import SparseReshapeDot

SPARSITY = 0.01
N_DATA = 100
BATCH_SIZE = 2
DIM_SPARSE = 2**16
N_EPOCH = 5

##################################################################################


def get_model():
    """ get sparse dot model """
    input_layer = Input(shape=(DIM_SPARSE,), sparse=True)
    input_c = Input(shape=(1,))
    
    e = input_c
    e = Embedding(input_dim=1, output_dim=1024)(e)
    def slice_reshape(x):
        return tf.reshape(tf.slice(x, begin=[0, 0, 0],
                                   size=[1, 1, 1024]), (1024, 1))
    e = Lambda(slice_reshape)(e)

    x = input_layer
    # (N, 32*2*1024) . (1024, 1) -> (N, 32, 2, 1)
    x = SparseReshapeDot(reshape=(32, 2, 1024))([x, e])
    x = Flatten()(x)
    output_layer = Dense(units=1)(x)

    model = Model([input_layer, input_c], output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

x = np.random.binomial(n=1, p=SPARSITY,
                       size=(N_DATA, DIM_SPARSE)).astype(np.float32)
x_sp = csr_matrix(x)
y = np.random.binomial(n=1, p=0.5, size=(N_DATA, 1)).astype(np.float32)

c = np.random.randint(0, 7, size=(N_DATA, 1)) / 7

def test_dot():
    """ test of dot model """
    model = get_model()
    model.fit([x_sp, c], y, batch_size=BATCH_SIZE, epochs=N_EPOCH)
    return True

if __name__ == '__main__':
    test_dot()


    

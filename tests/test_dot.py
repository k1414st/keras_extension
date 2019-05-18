import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input, Flatten, Embedding, Dot
from keras.models import Model
import tensorflow as tf

from keras.layers import Input, Flatten, Embedding, Dot
# from keras_extension.layers import Dense, SparseReshapeDense

SPARSITY = 0.01
N_DATA = 100
BATCH_SIZE = 2
DIM_SPARSE = 2**16
N_EPOCH = 2

##################################################################################


def get_model_fold():
    """ get sparse-adaptive fold model """
    input_layer = Input(shape=(2, 1024,))
    x = input_layer

    input_c = Input(shape=(1024,))
    # e = Embedding(input_dim=7, output_dim=1024)(input_c)
    e = input_c

    # x = SparseReshapeDense(reshape=(32, 2, 1024), units=10)(x)
    print(x.shape, e.shape)
    
    output_layer = Dot(axes=(1, 1))([x, e])
    print(output_layer.shape)
    # output_layer = Dense(units=1)(x)

    model = Model([input_layer, input_c], output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

# making dummy data (x is sparse matrix).
x = np.random.binomial(n=1, p=SPARSITY,
                       size=(N_DATA, 2, 1024)).astype(np.float32)
y = np.random.binomial(n=1, p=0.5, size=(N_DATA, 1)).astype(np.float32)

# c = np.random.randint(0, 7, size=(N_DATA,))
c = np.random.uniform(0, 1, size=(N_DATA, 1024,))

def test_dense_fold():
    """ test of not-sparsed fold-dense model """
    model = get_model_fold()
    model.fit([x, c], y, batch_size=BATCH_SIZE, epochs=N_EPOCH)
    return True


if __name__ == '__main__':
    test_dense_fold()

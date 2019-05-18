import numpy as np
from scipy.sparse.csr import csr_matrix
from keras.layers import Input, Flatten, Embedding, Dot, Dense
from keras.layers import Lambda, Reshape
from keras.models import Model
import tensorflow as tf

# from keras_extension.layers import Dense, SparseReshapeDense

SPARSITY = 0.01
N_DATA = 100
BATCH_SIZE = 2
DIM_SPARSE = 2**16
N_EPOCH = 2


#################################################################################

def get_model_fold():
    """ get sparse-adaptive fold model """
    input_layer = Input(shape=(1,))
    x = Embedding(input_dim=1, output_dim=1024, input_length=1)(input_layer)
    x = Flatten()(x)
    x = Dense(1)(x)
    output_layer = x

    model = Model(input_layer, output_layer)
    model.compile('adam', 'mse')
    model.summary()
    return model

# making dummy data (x is sparse matrix).
x = np.random.binomial(n=1, p=SPARSITY,
                       size=(N_DATA, DIM_SPARSE)).astype(np.float32)
x_sp = csr_matrix(x)
y = np.random.binomial(n=1, p=0.5, size=(N_DATA, 1)).astype(np.float32)

# c = np.random.randint(0, 7, size=(N_DATA, 1))
c = np.random.randint(0, 7, size=(100, 1)) / 7

def test_dense_fold():
    model = get_model_fold()
    model.fit(c, y, batch_size=BATCH_SIZE, epochs=N_EPOCH)
    return True


# real input dim must be 2.
# left matrix must be sparse, and reshaped.


if __name__ == '__main__':
    test_dense_fold()


    

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras_extension.layers import GraphConv

# constants of data shape.
# Number of data, Number of nodes, Dimension of input, Dimension of latent states.
N_data = 200
N_node = 100
D_input = 50
D_hidden = 10


def _get_model():
    """ get model using graph convolution layer. """

    input_layer = Input(shape=(N_node, D_input))  # L, D
    input_graph = Input(shape=(N_node, N_node))  # L, L
    g = GraphConv(D_hidden)

    x = g([input_layer, input_graph])
    output_layer = Dense(1, activation='tanh')(x)
    output_layer = Flatten()(output_layer)

    mdl = Model([input_layer, input_graph], output_layer)
    mdl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    mdl.summary()
    return mdl


def _get_simulated_data(random_state=None):
    """
    get simulated data
    shape[0] of all data means data size (N_data).

    X:  input node data (shape=(N_data, N_node, D_input)).
        X has each nodes informations with D_input size.
    G:  input graph data (shape=(N_data, N_node, N_node)).
        each data represents directed adjacency matrix.
    y:  answer data (shape=(N_data, N_node)).
        each data & each node has binary answers.
        (decided by sum of all input flow G.dot(X))
    """
    # set seed (by default, set time()).
    if random_state == None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(random_state)

    # simulating graphical data
    # Input node data
    X = np.random.uniform(0, 1, size=(N_data, N_node, D_input))

    # Graph as directed adjacency matrix
    G = np.random.binomial(n=1, p=0.1, size=(N_data, N_node, N_node))
    G = np.array([np.triu(g, 1) + np.tril(g, -1) for g in G])

    # If sum of all flow is more than 100, set flag = 1.
    y = np.einsum('ijk,ikl->ij', G, X)
    y = np.where(y > 250, 1, 0)

    return (X, G, y)


def test_graphconv():
    """ simple (not real, simulation based) performance test. """

    # get simulated data, and split to train & test.
    X, G, y = _get_simulated_data()
    X_train, X_test, G_train, G_test, y_train, y_test = \
        train_test_split(X, G, y, test_size=0.3)

    # model fit & predict
    mdl = _get_model()

    mdl.reset_states()
    mdl.fit([X_train, G_train], y_train,
            validation_data=([X_test, G_test], y_test),
            batch_size=32, epochs=30, verbose=1)
    y_test_pred = mdl.predict([X_test, G_test])
    auc = roc_auc_score(y_test.ravel(), y_test_pred.ravel())

    print('Final AUC:', auc)
    assert(auc > 0.7)



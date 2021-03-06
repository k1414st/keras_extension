import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.layers import Input, Flatten, Dense, GRUCell, LSTMCell
from keras.models import Model
from keras_extension.layers import GraphConv, GraphRNN, GraphRRNN
from keras.optimizers import Adam
from keras_extension.optimizers import AdaBound

# constants of data shape.
# Number of data, Number of nodes, Dimension of input, Dimension of latent states.
N_data = 200
N_node = 100
D_input = 50
D_hidden = 10

# number of test trial.
# (sometimes test failed perhaps because of simulation or initializer bias.)
N_TRIAL = 3

RANDOM_SEED_SIM = 9


def _get_model_wrapper(func_graph_layer):
    """
    get model wrapper using graph convolution layer.
    you need to implement only graph layer
    with this decorator function.
    """

    def wrapper():
        input_layer = Input(shape=(N_node, D_input))  # L, D
        input_graph = Input(shape=(N_node, N_node))  # L, L

        # definition of graph layer
        x = func_graph_layer(input_layer, input_graph)

        output_layer = Dense(1, activation='tanh')(x)
        output_layer = Flatten()(output_layer)

        mdl = Model([input_layer, input_graph], output_layer)
        # opt = Adam()
        opt = AdaBound(terminal_bound=0.01)
        mdl.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        mdl.summary()
        return mdl

    return wrapper


@_get_model_wrapper
def _get_model_graphconv(input_layer, input_graph):
    """ get model using GraphConv. """
    g = GraphConv(units=D_hidden)
    return g([input_layer, input_graph])


@_get_model_wrapper
def _get_model_gate(input_layer, input_graph):
    """ get model using GraphConv. """
    g = GraphConv(units=D_hidden, gate_units=4)
    return g([input_layer, input_graph])

@_get_model_wrapper
def _get_model_gat(input_layer, input_graph):
    """ get model using GraphConv. """
    g = GraphConv(units=D_hidden, gat_units=16, gat_n_heads=3)
    return g([input_layer, input_graph])


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
        np.random.seed(int((time.time() % 100)*100))
    else:
        np.random.seed(random_state)

    # simulating graphical data.
    # Input node data
    X = np.random.uniform(0, 1, size=(N_data, N_node, D_input))

    # Graph as directed adjacency matrix (binary).
    G = np.random.binomial(n=1, p=0.1, size=(N_data, N_node, N_node))
    # remove diagonal elements.
    G = np.array([np.triu(g, 1) + np.tril(g, -1) for g in G])

    # custom sum of input flow. (condition sum)
    # If sum of all flow is more than median, set flag = 1.
    X_cond = np.expand_dims(X[:, :, -1] > 0.5, axis=2)
    y = np.einsum('ijk,ikl->ij', G, X_cond * X)
    y = np.where(y > np.median(y), 1, 0)

    return (X, G, y)


def _test_graph_model(get_model_function):
    """ simple (not real, simulation based) performance test. """

    list_auc = []
    for i in range(N_TRIAL):
        # get simulated data, and split to train & test.
        X, G, y = _get_simulated_data(RANDOM_SEED_SIM + i)
        X_train, X_test, G_train, G_test, y_train, y_test = \
            train_test_split(X, G, y, test_size=0.3)

        # model fit & predict
        mdl = get_model_function()

        mdl.fit([X_train, G_train], y_train,
                validation_data=([X_test, G_test], y_test),
                batch_size=32, epochs=500, verbose=1)
        y_test_pred = mdl.predict([X_test, G_test])
        auc = roc_auc_score(y_test.ravel(), y_test_pred.ravel())

        print('AUC:', auc)
        list_auc.append(auc)

    # Any trial get over 0.7.
    assert((np.array(list_auc) > 0.7).any())


def test_graphconv():
    _test_graph_model(_get_model_graphconv)


def test_gate():
    _test_graph_model(_get_model_gate)


def test_gat():
    _test_graph_model(_get_model_gat)


if __name__ == '__main__':
    test_gate()

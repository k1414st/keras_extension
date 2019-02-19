# -*- coding: utf-8 -*-

"""
Graph Neural Network layers.
This implementation is based on
"GraphIE: A Graph-Based Framework for Information Extraction"
(https://arxiv.org/abs/1810.13083)
"Graph Convolutional Networks for Text Classification"
(https://arxiv.org/abs/1809.05679)
"""

import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.backend import backend


def _batch_dot(x, y, axes=None):
    """
    K.batch_dot is only limited in the case of (x.ndim<=3 and y.ndim<=3).
    this wrapped function is not limited to that case.
    (using "tf.einsum" or "K.batch_dot with reshape".)
    """
    if K.ndim(x) <= 3 and K.ndim(y) <= 3:
        return K.batch_dot(x, y, axes=axes)
    else:
        if (axes[0] == 0) or (axes[1] == 0):
            raise ValueError('each axis must not be 0 (N_batch place).')
        if backend() == 'tensorflow':
            if K.ndim(x) + K.ndim(y) >= 27:
                raise ValueError('x.ndim + y.ndim must be less than 27')
            import tensorflow as tf

            # make einsum string
            # ex). NabcXefg, NhijkXlm -> Nabcefghijklm (N: pos_batch, X: pos_axes)
            str_x = ''.join([chr(97+i) for i in range(K.ndim(x))])
            str_y = ''.join([chr(97+len(str_x)+i) for i in range(K.ndim(y))])
            str_y = str_y.replace(str_y[0], str_x[0])
            str_y = str_y.replace(str_y[axes[1]], str_x[axes[0]])
            str_out = str_x.replace(str_x[axes[0]], '') + \
                str_y.replace(str_y[axes[1]], '')[1:]
            str_einsum = '%s,%s->%s' % (str_x, str_y, str_out)

            return tf.einsum(str_einsum, x, y)
        else:
            # set shape, targat-idim, target-shape
            sx, sy = x.shape, y.shape
            ax0, ax1 = axes[0], axes[1]
            s0, s1 = sx[ax0], sy[ax1]

            # reshape: (B, a1, a2, ... axis, ... an) -> (B, prod(~axis), axis)
            dx_rm = [i for i in range(len(sx)) if i != ax0]
            dy_rm = [i for i in range(len(sy)) if i != ax1]
            sx_rm = [sx[i] for i in dx_rm]
            sy_rm = [sy[i] for i in dy_rm]
            x = K.permute_dimensions(x, dx_rm + [ax0])
            y = K.permute_dimensions(y, dy_rm + [ax1])
            x = K.reshape(x, [-1, np.prod(sx_rm[1:]), s0])
            y = K.reshape(y, [-1, np.prod(sy_rm[1:]), s1])

            # reshape: (B, prod(sx_rm), prod(sy_rm)) -> (B, sx_rm, sy_rm)
            out = K.batch_dot(x, y, axes=(K.ndim(x)-1, K.ndim(y)-1))
            return K.reshape(out, [-1] + sx_rm[1:] + sy_rm[1:])


class _ParametricLayer(Layer):
    """
    Layer class for adding trainable parameters.
    This class provide protected method _add_w &  _add_b.

    Init Args:
        bias_initializer: Initializer for the bias vector
        bias_regularizer: Regularizer function applied to the bias vector
        bias_constraint: Constraint function applied to the bias vector
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix

        see [https://keras.io/initializers], [https://keras.io/regularizers],
            [https://keras.io/constraints]
    """

    def __init__(self,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(_ParametricLayer, self).__init__(**kwargs)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def _add_w(self, shape, name):
        return self.add_weight(shape=shape, name=name+'_weight',
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)

    def _add_b(self, shape, name):
        return self.add_weight(shape=shape, name=name+'_bias',
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)


class GraphConv(_ParametricLayer):
    """
    Graphical Convolution Layer connected by user-specified weighted-digraph.
    You must input graph-data node-data.

    Args:
        units: Positive integer, dimensionality of the output space.
        use_node_weight: use graph-node self-loop weight.
            if False, no special self-loop weight is added.
            (diagonal component of graph-egde is used as self-loop implicitly)
        activation: Activation function of output.
            default: 'sigmoid'
        use_bias: use bias vector or not.

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 units,
                 use_node_weight=True,
                 activation='sigmoid',
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(GraphConv, self).__init__(
            bias_initializer, bias_regularizer, bias_constraint,
            kernel_initializer, kernel_regularizer, kernel_constraint,
            **kwargs)
        self.units = units
        self.use_node_weight = use_node_weight
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shapes):
        # input: (N_batch, L, D),  graph: (N_batch, L, L)
        # L: node_size
        # D: feat_size (self.units)
        self.node_size = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.e_weight = self._add_w((input_size, self.units), 'e')
        if self.use_node_weight:
            self.v_weight = self._add_w((input_size, self.units), 'v')
        if self.use_bias:
            self.bias = self._add_b((self.units,), 'all')
        self.built = True

    def call(self, inputs, training=None):
        """
        Args:
            input[0]: input_layer(N_Batch, L_sequence, Dim_fature)
            input[1]: weighted-digraph(L, L) = (from, to)
        Return:
            output_layer(N_Batch, L_sequence, Dim_feature)
        """
        seq_data = inputs[0]
        graph = inputs[1]

        # beta (edge)
        beta = K.dot(seq_data, self.e_weight)
        beta = K.batch_dot(graph, beta, axes=(2, 1))  # BL(o)L(i),BL(i)D,->BL(o)D

        # connect edge, (node), bias
        out = beta
        if self.use_bias:
            out = K.bias_add(out, self.bias)
        if self.use_node_weight:
            alpha = K.dot(seq_data, self.v_weight)
            out = out + alpha
        gi = self.activation(out)
        return gi

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.units)


class MultiGraphConv(_ParametricLayer):
    """
    Graphical Convolution Layer using layered graphs.
    You must input graph-data node-data.

    Args:
        units: Positive integer, dimensionality of the output space.
        use_node_weight: use graph-node self-loop weight.
            if False, no special self-loop weight is added.
            (diagonal component of graph-egde is used as self-loop implicitly)
        activation: Activation function of output.
            default: 'sigmoid'
        use_bias: use bias vector or not.

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 units,
                 use_node_weight=True,
                 activation='sigmoid',
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(MultiGraphConv, self).__init__(
            bias_initializer, bias_regularizer, bias_constraint,
            kernel_initializer, kernel_regularizer, kernel_constraint,
            **kwargs)
        self.units = units
        self.use_node_weight = use_node_weight
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shapes):
        # input: (N_batch, L, D),  graph: (N_batch, L, L, M)
        # L: node_size
        # D: feat_size (self.units)
        # M: graph multi size
        self.node_size = input_shapes[0][-2]
        input_size = input_shapes[0][-1]
        multi_size = input_shapes[1][-1]

        self.e_weight = self._add_w((input_size, self.units), 'e')
        if self.use_node_weight:
            self.v_weight = self._add_w((input_size, multi_size, self.units), 'v')
        if self.use_bias:
            self.bias = self._add_b((self.units, multi_size), 'all')
        self.built = True

    def call(self, inputs, training=None):
        """
        Args:
            input[0]: input_layer(N_Batch, L_sequence, Dim_fature)
            input[1]: weighted-digraph(L, L) = (from, to)
        Return:
            output_layer(N_Batch, L_sequence, Dim_feature)
        """
        seq_data = inputs[0]
        graph = inputs[1]

        # beta (edge)
        beta = K.dot(seq_data, self.e_weight)
        beta = _batch_dot(graph, beta, axes=(2, 1))  # BL(o)L(i)M,BL(i)D,->BL(o)MD

        # connect edge, (node), bias
        out = beta
        if self.use_bias:
            out = K.bias_add(out, self.bias)
        if self.use_node_weight:
            s = self.v_weight.shape
            w = K.reshape(self.v_weight, (s[0], s[1]*s[2]))
            alpha = K.dot(seq_data, w)
            alpha = K.reshape(alpha, (-1, alpha.shape[1], s[1], s[2]))
            out = out + alpha
        gi = self.activation(out)
        return gi

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][-1], self.units)


class GraphRNN(_ParametricLayer):
    """
    Graph Recurrent Network Layer connected by user-specified weighted-digraph.
    when creating object, you can choose recurrent cell (LSTMCell, GRUCell, etc).

    Args:
        cell: A RNN cell instance. A RNN cell is a class that has
            call method and state_size attribute.
        activation: Activation function of output.
            default: 'sigmoid'

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 cell,
                 activation='sigmoid',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(GraphRNN, self).__init__(
            bias_initializer, bias_regularizer, bias_constraint,
            kernel_initializer, kernel_regularizer, kernel_constraint,
            **kwargs)
        self.cell = cell
        self.activation = activations.get(activation)

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def build(self, input_shapes):
        # input: (N_batch, L, D),  graph: (N_batch, L, L)
        self.node_size = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.e_weight = self._add_w((input_size, self.cell.units), 'e')
        self.cell.build((input_shapes[0][0], self.cell.units))
        self.built = True

    def call(self, inputs, initial_state=None, training=None):
        """
        Args:
            input[0]: input_layer(N_Batch, L_sequence, Dim_fature)
            input[1]: weighted-digraph(L, L) = (from, to)
        Return:
            output_layer(N_Batch, L_sequence, Dim_feature)
        """

        if initial_state is not None:
            pass
        else:
            initial_state = self.get_initial_state(inputs[0])

        seq_data = inputs[0]
        graph = inputs[1]

        # beta (edge)
        beta = K.dot(seq_data, self.e_weight)
        g_beta = K.batch_dot(graph, beta, axes=(2, 1))  # BL(o)L(i),BL(i)D,->BL(o)D
        print(beta.shape)

        # last_output, outputs, states = \
        #     K.rnn(lambda inputs, states: self.cell.call(inputs, states),
        #           beta,
        #           initial_state)
        x = self.cell.call(g_beta, beta)
        return x[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.cell.units)




from keras.layers import Input, Flatten, Dense, GRUCell
from keras.models import Model
# from keras_extension.layers import GraphConv, GraphRNN

# constants of data shape.
# Number of data, Number of nodes, Dimension of input, Dimension of latent states.
N_data = 200
N_node = 100
D_input = 50
D_hidden = 10



input_layer = Input(shape=(N_node, D_input))  # L, D
input_graph = Input(shape=(N_node, N_node))  # L, L
g = GraphRNN(GRUCell(units=D_hidden))


x = g([input_layer, input_graph])

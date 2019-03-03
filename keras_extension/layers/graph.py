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

    def _graph_attention(self, g, x, w, a):
        """
        using graph attention mechanism.

        Args:
            g: input Tensor of graph adjacency matrix.
               shape: (B(Batch_size), N(N_nodes), N)
            x: input Tensor of node-data after convolutioned.
               shape: (B, N, F_in(F_inputs))
            w: weight matrix variable
               (to transform input to attentionable hidden states.)
               shape: (F_in, F(F_outputs) * H(N_heads))
            a: merge weight vector from attentionable state to attention value.
               shape: (2 * F,)
        """
        F_in, F, H = w.shape[0], w.shape[1], w.shape[2]
        N = g.shape[-1]

        # w = K.reshape(F1, H * F2)  # (F_in, H*F)
        x = K.expand_dims(K.dot(x, w), axis=1)  # (B, 1, H*F)
        x = K.concatenate([x[:, :, F*i:F*(i+1)]
                            for i in range(H)], axis=1)  # (B, H, F)

        # concat meshly
        _x1 = K.tile(K.expand_dims(XX, axis=0), (N, 1, 1, 1))
        _x2 = K.tile(K.expand_dims(XX, axis=1), (1, N, 1, 1))
        x = K.concatenate([_x1, _x2], axis=3)  # (N, N, H, 2F)

        def _expand_dims_recursive(x, axis_list):
            assert(len(axis_list) > 0)
            if len(axis_list) == 1:
                return K.expand_dims(x, axis_list[0])
            return _expand_dims_recursive(K.expand_dims(x, axis_list[0]),
                                          axis_list=axis_list[1:])
        # squeeze 2F
        a = _expand_dims_recursive(a, (0, 0, 0))
        x = activation(K.sum(x * a, axis=-1))  # (N, N, H)

        # normalize by neighbors
        x_norm = K.sum(x * K.expand_dims(g, axis=2),
                    axis=1, keepdims=True)  # (N, 1, H)
        return x / x_norm  # (N, N, H)


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
        cell: Set a RNN cell instance. A RNN cell is a class that must have
            call method and state_size attribute.
        return_states: return states
            if True, 
            call method and state_size attribute.
        activation: Activation function of output.
            default: 'sigmoid'

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 cell,
                 return_states=False,
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
        self.return_states = return_states
        self.activation = activations.get(activation)

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        # reshape to (N, nodes, input_dim) -> (N, nodes, hidden_dim)
        initial_state = K.zeros_like(inputs)[:, :, 0]
        initial_state = K.expand_dims(initial_state, axis=2)
        return K.tile(initial_state, [1, 1, self.cell.units])

    def build(self, input_shapes):
        # input: (N_batch, L, D),  graph: (N_batch, L, L)
        self.node_size = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.e_weight = self._add_w((input_size, self.cell.units), 'e')
        self.cell.build((input_shapes[0][0], self.cell.units))
        self.built = True

    def call(self, inputs, encode=True, training=None):
        """
        Args:
            input[0]: input_layer(N_Batch, L_sequence, Dim_fature)
            input[1]: weighted-digraph(L, L) = (from, to)
        Return:
            output_layer(N_Batch, L_sequence, Dim_feature)
        """
        if training is not None:
            raise NotImplementedError('training option is not implemented yet.')

        input_data = inputs[0]
        graph = inputs[1]

        if len(inputs) == 3:
            state = inputs[2]
        else:
            state = self.get_initial_state(inputs[0])

        if encode:
            beta = K.dot(input_data, self.e_weight)
        else:
            beta = input_data

        # BL(o)L(i),BL(i)D,->BL(o)D
        agg_beta = K.batch_dot(graph, beta, axes=(2, 1))
        # output = (h, [h, c])
        outputs, states = self.cell.call(beta, [agg_beta, state])

        if self.return_states:
            return [outputs, states[1]]
        else:
            return outputs

    def compute_output_shape(self, input_shape):
        if self.return_states:
            return [(input_shape[0][0], input_shape[0][1], self.cell.units),
                    (input_shape[0][0], input_shape[0][1], self.cell.units)]
        else:
            return (input_shape[0][0], input_shape[0][1], self.cell.units)


class GraphTimeSeriesRNN():
    """
    """
    def __init__():
        raise NotImplementedError('GraphTimeSeriesRNN is not implemented yet.')

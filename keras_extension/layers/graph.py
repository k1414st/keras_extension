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
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

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
                 gate_units=None,
                 gate_mode=None,
                 gat_units=None,
                 gat_n_heads=None,
                 **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.use_node_weight = use_node_weight
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # gate
        if gate_units is not None:
            if gate_mode is None:
                gate_mode = 'dual_weight'
            if gate_mode in ['single_weight', 'dual_weight']:
                self.use_gate = True
                self.gate_units = gate_units
                self.gate_mode = gate_mode
            else:
                raise ValueError('set gate_mode to "(single|dual)_weight".')
        else:
            self.use_gate = False
        # gat
        if gat_units is not None:
            if gat_n_heads is not None:
                self.use_gat = True
                self.gat_units = gat_units
                self.gat_n_heads = gat_n_heads
            else:
                raise ValueError('set gat_units & gat_n_heads simultaneously.')
        else:
            self.use_gat = False

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
        if self.use_gate:
            self.gate_w1 = \
                self._add_w((input_size, self.gate_units), 'gate_w1')
            if self.gate_mode == 'dual_weight':
                self.gate_w2 = \
                    self._add_w((input_size, self.gate_units), 'gate_w2')
        if self.use_gat:
            self.att_w_weight = \
                self._add_w((input_size, self.gat_units*self.gat_n_heads), 'att_w')
            self.att_a_weight = self._add_w((self.gat_units*2,), 'att_a')
        self.built = True

    def __graph_attention(self, g, x, w, a, n_heads=1):
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
        H = n_heads
        F_in, FH = w.shape[0], w.shape[1]
        F = FH // H
        N = g.shape[-1]

        # w = K.reshape(F1, H * F2)  # (F_in, H*F)
        x = K.expand_dims(K.dot(x, w), axis=2)  # (B, N, 1, H*F)
        x = K.concatenate([x[:, :, :, F*i:F*(i+1)]
                           for i in range(H)], axis=2)  # (B, N, H, F)

        # concat meshly
        _x1 = K.tile(K.expand_dims(x, axis=1), (1, N, 1, 1, 1))
        _x2 = K.tile(K.expand_dims(x, axis=2), (1, 1, N, 1, 1))
        x = K.concatenate([_x1, _x2], axis=4)  # (B, N, N, H, 2F)

        def _expand_dims_recursive(x, axis_list):
            assert(len(axis_list) > 0)
            if len(axis_list) == 1:
                return K.expand_dims(x, axis_list[0])
            return _expand_dims_recursive(K.expand_dims(x, axis_list[0]),
                                          axis_list=axis_list[1:])
        # squeeze 2F
        a = _expand_dims_recursive(a, (0, 0, 0, 0))
        x = K.exp(K.relu(K.sum(x * a, axis=-1), alpha=0.2))  # (B, N, N, H)

        # normalize by neighbors
        x_norm = K.sum(x * K.expand_dims(g, axis=-1),
                       axis=2, keepdims=True)  # (B, N, 1, H)
        return x / x_norm  # (B, N, N, H)

    def __graph_gate(self, x, w1, w2=None):
        """
        make a graph gate matrix by crossing each nodes latent states.

        Args:
            x: input Tensor of node-data after convolutioned.
               shape: (B(Batch_size), N(N_nodes), F_in(F_inputs))
            w1, w2: weight matrix variable
               (to transform input to gatable hidden states.)
               shape: (F_in, F_out(F_outputs))
        """
        if w2 is None:
            w2 = w1
        h1 = K.dot(x, w1)  # (B, N, F_out)
        h2 = K.dot(x, w2)  # (B, N, F_out)
        hh = K.batch_dot(h1, h2, axes=(2, 2))  # (B, N, N)
        return K.sigmoid(hh)  # (B, N, N)

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

        # graph gate
        if self.use_gate:
            if self.gate_mode == 'single_weight':
                graph = graph * self.__graph_gate(seq_data, self.gate_w1)
            else:
                graph = graph * \
                    self.__graph_gate(seq_data, self.gate_w1, self.gate_w2)

        # beta (edge)
        beta = K.dot(seq_data, self.e_weight)
        if self.use_gat:
            att_alpha = \
                self.__graph_attention(graph, seq_data, self.att_w_weight,
                                       self.att_a_weight, n_heads=self.gat_n_heads)
            list_att_beta = []
            for i in range(self.gat_n_heads):
                att_beta = K.batch_dot(att_alpha[:, :, :, i], beta, axes=(2, 1))
                list_att_beta.append(att_beta)
            beta = sum(list_att_beta)
        else:
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


class GraphRNN(_ParametricLayer):
    """
    Graph Recurrent Network Layer connected by user-specified weighted-digraph.
    when creating object, you can choose recurrent cell (LSTMCell, GRUCell, etc).

    Args:
        cell: Set a RNN cell instance. A RNN cell is a class that must have
            call method and state_size attribute.
        return_state: return states or not.
        activation: Activation function of output.
            default: 'sigmoid'

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 cell,
                 return_state=False,
                 activation='sigmoid',
                 **kwargs):
        super(GraphRNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_state = return_state
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

        if self.return_state:
            return [outputs, states[1]]
        else:
            return outputs

    @property
    def trainable_weights(self):
        return [self.e_weight] + self.cell.trainable_weights

    def compute_output_shape(self, input_shape):
        if self.return_state:
            return [(input_shape[0][0], input_shape[0][1], self.cell.units),
                    (input_shape[0][0], input_shape[0][1], self.cell.units)]
        else:
            return (input_shape[0][0], input_shape[0][1], self.cell.units)


class GraphRRNN(_ParametricLayer):
    """
    Graph Recurrent and Time Recurrent Network Layerconnected by
    user-specified weighted-digraph.
    when creating object, you can choose recurrent cell (LSTMCell, GRUCell, etc).

    Args:
        cell: Set a RNN cell instance. A RNN cell is a class that must have
            call method and state_size attribute.
        return_state: return states
        activation: Activation function of output.
            default: 'sigmoid'

        (bias |kernel)_(initializer |regularizer |constraint):
            see [https://keras.io/initializers], [https://keras.io/regularizers],
                [https://keras.io/constraints]
    """

    def __init__(self,
                 cell,
                 n_layers,
                 return_sequences=False,
                 output_sequence_axis=-1,
                 return_state=False,
                 activation='sigmoid',
                 **kwargs):
        super(GraphRRNN, self).__init__(**kwargs)
        self.cell = cell
        self.n_layers = n_layers
        self.return_sequences = return_sequences
        self.output_sequence_axis = output_sequence_axis
        self.return_state = return_state
        self.activation = activations.get(activation)
        self.grnn_layer = \
            GraphRNN(cell=cell,
                     return_state=True,
                     activation=activation,
                     **kwargs)

    def build(self, input_shapes):
        self.grnn_layer.build(input_shapes)
        self.built = True

    def call(self, inputs):
        """
        Args:
            input[0]: input_layer(N_Batch, L_sequence, Dim_fature)
            input[1]: weighted-digraph(L, L) = (from, to)
        Return:
            output_layer(N_Batch, L_sequence, Dim_feature)
        """
        input_data = inputs[0]
        graph = inputs[1]

        xs = [None] * self.n_layers
        ss = [None] * self.n_layers

        xs[0], ss[0] = self.grnn_layer.call([input_data, graph])
        for i in range(1, self.n_layers):
            xs[i], ss[i] = \
                self.grnn_layer.call([xs[i-1], graph, ss[i-1]], encode=False)
        return xs[self.n_layers-1]

        # return values depend on options.
        if self.return_sequences:
            xs = [K.expand_dims(x, axis=self.output_sequence_axis) for x in xs]
            ss = [K.expand_dims(s, axis=self.output_sequence_axis) for s in ss]
            if self.return_state:
                return (K.concatenate(xs, axis=self.output_sequence_axis),
                        K.concatenate(xs, axis=self.output_sequence_axis))
            else:
                return K.concatenate(xs, axis=self.output_sequence_axis)
        else:
            if self.return_state:
                return xs[-1], ys[-1]
            else:
                return xs[-1]

    @property
    def trainable_weights(self):
        return self.grnn_layer.trainable_weights

    def compute_output_shape(self, input_shape):
        shape_1 = (input_shape[0][0], input_shape[0][1], self.grnn_layer.cell.units)
        shape_all = list(shape_1)
        shape_all.insert(self.output_sequence_axis, self.n_layers)

        if self.return_sequences:
            if self.return_state:
                return [shape_all, shape_all]
            else:
                return shape_all
        else:
            if self.return_state:
                return [shape_1, shape_1]
            else:
                return shape_1

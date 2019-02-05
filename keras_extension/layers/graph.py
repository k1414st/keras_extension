# -*- coding: utf-8 -*-

"""
Recurrent layers and their base classes.
This implementation is based on
"GraphIE: A Graph-Based Framework for Information Extraction"
(https://arxiv.org/abs/1810.13083)
"Graph Convolutional Networks for Text Classification"
(https://arxiv.org/abs/1809.05679)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer


class GraphConv(Layer):
    """
    Graphical Convolution Layer from sequenttal to sequential
    connected by user-specified (static) weighted-digraph.
    Graph is dynamic, you must input graph-data correspond to your
    input sequential data.

    Args:
        units: Positive integer, dimensionality of the output space.
        use_vertex_weight: use graph-vertex self-loop weight.
            if False, no special self-loop weight is added.
            (diagonal component of graph-egde is used as self-loop implicitly)
        normalize_graph_input: normalize graph input.
            sum of graph-arrow input to one node is normalized to 1.
        normalize_graph_output: normalize graph output.
            sum of graph-arrow output from one node is normalized to 1.
        activation: Activation function of output.
            default: 'sigmoid'

        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 units,
                 use_vertex_weight=True,
                 normalize_graph_input=False,
                 normalize_graph_output=True,
                 activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.use_vertex_weight = use_vertex_weight
        self.normalize_graph_input = normalize_graph_input
        self.normalize_graph_output = normalize_graph_output
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

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

    def build(self, input_shapes):
        # graph: (N_batch, L, L),  input: (N_batch, L, D)
        self.length = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.v_weight = self._add_w((input_size, self.units), 'v')
        self.e_weight = self._add_w((input_size, self.units), 'e')
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

        ### beta (edge)
        beta = K.dot(seq_data, self.e_weight)

        # normalize graph (input-base or output-base)
        if self.normalize_graph_output:
            # Normalize factor of graph-arrow input (B, L(out))
            norm = K.maximum(K.sum(graph, axis=2), K.epsilon())
            beta = beta / K.expand_dims(norm, axis=2)  # BL(in)D,BL(in)1->BL(in)D
        if self.normalize_graph_input:
            # Normalize factor of graph-arrowinput (B, L(in))
            norm = K.maximum(K.sum(graph, axis=1), K.epsilon())
            beta = beta / K.expand_dims(norm, axis=1)  # BDL(o),B1L(o)->BDL(o)

        beta = K.batch_dot(beta, graph, axes=(1, 1))  # BLD,BL(i)L(o)->BDL(o)
        beta = K.permute_dimensions(beta, (0, 2, 1))  # BDL(o)->BL(o)D

        # connect edge, (vertex), bias
        if self.use_vertex_weight:
            alpha = K.dot(seq_data, self.v_weight)
            gi = self.activation(K.bias_add(alpha + beta, self.bias))
        else:
            gi = self.activation(K.bias_add(beta, self.bias))
        return gi

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.units)


class GraphRNN(Layer):
    """
    Graphical Convolution Layer from sequenttal to sequential
    connected by user-specified (static) weighted-digraph.
    Graph is dynamic, you must input graph-data correspond to your
    input sequential data.

    Args:
        cell: A RNN cell instance. A RNN cell is a class that has
            call method and state_size attribute.
        normalize_graph_input: normalize graph input.
            sum of graph-arrow input to one node is normalized to 1.
        normalize_graph_output: normalize graph output.
            sum of graph-arrow output from one node is normalized to 1.
        activation: Activation function of output.
            default: 'sigmoid'

        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 cell,
                 normalize_graph_input=False,
                 normalize_graph_output=True,
                 activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphRNN, self).__init__(**kwargs)
        self.cell = cell
        self.normalize_graph_input = normalize_graph_input
        self.normalize_graph_output = normalize_graph_output
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

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
        # graph: (N_batch, L, L),  input: (N_batch, L, D)
        self.length = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.e_weight = self._add_w((input_size, self.cell.units), 'e')
        # self.cell.build((input_shapes[0][0], input_shapes[0][2]))
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

        ### beta (edge)
        beta = K.dot(seq_data, self.e_weight)

        # normalize graph (input-base or output-base)
        if self.normalize_graph_output:
            # Normalize factor of graph-arrow input (B, L(out))
            norm = K.maximum(K.sum(graph, axis=2), K.epsilon())
            beta = beta / K.expand_dims(norm, axis=2)  # BL(in)D,BL(in)1->BL(in)D
        if self.normalize_graph_input:
            # Normalize factor of graph-arrowinput (B, L(in))
            norm = K.maximum(K.sum(graph, axis=1), K.epsilon())
            beta = beta / K.expand_dims(norm, axis=1)  # BDL(o),B1L(o)->BDL(o)

        beta = K.batch_dot(beta, graph, axes=(1, 1))  # BLD,BL(i)L(o)->BDL(o)
        beta = K.permute_dimensions(beta, (0, 2, 1))  # BDL(o)->BL(o)D

        def step(inputs, states):
            return self.cell.call(inputs, states)

        last_output, outputs, states = \
            K.rnn(step, beta,
                  initial_state,
                  input_length=self.length)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.cell.units)


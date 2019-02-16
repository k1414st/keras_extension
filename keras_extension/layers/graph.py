# -*- coding: utf-8 -*-

"""
Graph Neural Network layers.
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
    Graphical Convolution Layer connected by user-specified weighted-digraph.
    You must input graph-data node-data.

    Args:
        units: Positive integer, dimensionality of the output space.
        use_vertex_weight: use graph-vertex self-loop weight.
            if False, no special self-loop weight is added.
            (diagonal component of graph-egde is used as self-loop implicitly)
        activation: Activation function of output.
            default: 'sigmoid'

        use_bias: use bias vector or not.
        bias_initializer: Initializer for the bias vector
        bias_regularizer: Regularizer function applied to the bias vector
        bias_constraint: Constraint function applied to the bias vector
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
    """

    def __init__(self,
                 units,
                 use_vertex_weight=True,
                 activation='sigmoid',
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.use_vertex_weight = use_vertex_weight
        self.activation = activations.get(activation)
        self.use_bias = use_bias
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

    def build(self, input_shapes):
        # graph: (N_batch, L, L),  input: (N_batch, L, D)
        self.length = input_shapes[0][-2]
        input_size = input_shapes[0][-1]

        self.e_weight = self._add_w((input_size, self.units), 'e')
        if self.use_vertex_weight:
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

        ### beta (edge)
        beta = K.dot(seq_data, self.e_weight)
        beta = K.batch_dot(graph, beta, axes=(1, 1))  # BL(i)L(o),BL(i)D,->BL(o)D

        # connect edge, (vertex), bias
        out = beta
        if self.use_bias:
            out = K.bias_add(out, self.bias)
        if self.use_vertex_weight:
            alpha = K.dot(seq_data, self.v_weight)
            out = out + alpha
        gi = self.activation(out)
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
        activation: Activation function of output.
            default: 'sigmoid'

        bias_initializer: Initializer for the bias vector
        bias_regularizer: Regularizer function applied to the bias vector
        bias_constraint: Constraint function applied to the bias vector
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
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
        super(GraphRNN, self).__init__(**kwargs)
        self.cell = cell
        self.activation = activations.get(activation)
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
        beta = K.batch_dot(graph, beta, axes=(1, 1))  # BL(i)L(o),BL(i)D,->BL(o)D

        last_output, outputs, states = \
            K.rnn(lambda inputs, states: self.cell.call(inputs, states),
                  beta,
                  initial_state,
                  input_length=self.length)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.cell.units)

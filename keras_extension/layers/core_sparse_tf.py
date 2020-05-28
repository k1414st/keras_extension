# -*- coding: utf-8 -*-
"""
Customized Core Keras layers.
for tensorflow backend.
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.merge import _Merge

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

from .core_sparse import SparsableLayer


class Dense(SparsableLayer):
    """Just your regular densely-connected NN layer.

    This code is mostly copied from original Lambda class,
    but modified to be adaptable to input SparseTensor.


    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sparse=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(
                activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        if sparse:
            self.input_spec = InputSpec()
        else:
            self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if self._is_sparse:
            self.input_spec = InputSpec(axes={-1: last_dim})
        else:
            self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
            name='kernel',
            shape=(last_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        if self._is_sparse:
            output = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        else:
            output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseReshapeDense(Dense):
    """
    Extention for folded 2D sparse input multiplication.
    If input shape is represented as (batch_size, prod(reshape)),
    reshape the input to (batch_size, *reshape), and
    apply Dense Unit.

    Added Args:
        reshape: folded shape (reshape input by this).
    Returns:
        Densed Tensor: (batch_size, *reshape[-1:], dense.units)
    """

    def __init__(self, units,
                 reshape,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sparse=None,
                 **kwargs):
        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         sparse=True,
                         **kwargs)
        self.reshape = reshape

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if not self._is_sparse:
            raise ValueError('input must be 2d sparse matrix.')
        
        input_dim = self.reshape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self._is_sparse:
            self.input_spec = InputSpec(axes={-1: input_shape[-1]})
        else:
            self.input_spec = InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
        self.built = True

    def call(self, inputs):
        output = super().call(tf.sparse.reshape(inputs, (-1, self.reshape[-1])))
        output_shape = [-1] + list(self.reshape)
        output_shape[-1] = self.units
        output = tf.reshape(output, output_shape)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = [None] + list(self.reshape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class SparseReshapeDot(_Merge):
    """
    Layer that computes a dot product between
    pseudo multi-dimensional sparse_tensor and 2d dense matrix.
    sparse_tensor internally has shape (N, a, b, ..., k),
    but must be reshaped (N, a*b*...*k) beforehand.

    restriction:
        1. dense_matrix must be 2-dim.
        2. batch_dot option is not implemented yet.

    Args:
        reshape: internal shape (a, b, ..., k)
    """

    def __init__(self, reshape, **kwargs):
        super(SparseReshapeDot, self).__init__(**kwargs)
        self.reshape = reshape
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        # n(Ij).jk -> nIk
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        assert(len(shape1) == 2)
        assert(len(shape2) == 2)
        self.__output_shape \
            = tuple([None] + list(self.reshape[:-1]) + [shape2[-1]])


    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on exactly 2 inputs')
        x1 = inputs[0]
        x2 = inputs[1]
        x1 = tf.sparse.reshape(x1, (-1, self.reshape[-1]))
        output = tf.sparse.sparse_dense_matmul(x1, x2)
        output = K.reshape(output, [-1] + list(self.__output_shape[1:]))
        return output


    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        return tuple(self.__output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes,
        }
        base_config = super(SparseReshapeDot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


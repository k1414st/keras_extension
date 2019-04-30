# -*- coding: utf-8 -*-
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import copy
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.utils import conv_utils
from keras.legacy import interfaces


class Lambda(Layer):
    """Wraps arbitrary expression as a `Layer` object.

    This code is mostly copied from original Lambda class,
    but modified to be adaptable to input SparseTensor.

    # Examples

    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        model.add(Lambda(antirectifier,
                         output_shape=antirectifier_output_shape))
    ```

    # Arguments
        function: The function to be evaluated.
            Takes input tensor as first argument.
        output_shape: Expected output shape from function.
            Only relevant when using Theano.
            Can be a tuple or function.
            If a tuple, it only specifies the first dimension onward;
                 sample dimension is assumed either the same as the input:
                 `output_shape = (input_shape[0], ) + output_shape`
                 or, the input is `None` and
                 the sample dimension is also `None`:
                 `output_shape = (None, ) + output_shape`
            If a function, it specifies the entire shape as a function of the
            input shape: `output_shape = f(input_shape)`
        arguments: optional dictionary of keyword arguments to be passed
            to the function.

    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Specified by `output_shape` argument
        (or auto-inferred when using TensorFlow or CNTK).
    """

    @interfaces.legacy_lambda_support
    def __init__(self, function, output_shape=None,
                 mask=None, arguments=None, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.function = function
        self.arguments = arguments if arguments else {}
        if mask is not None:
            self.supports_masking = True
        self.mask = mask

        if output_shape is None:
            self._output_shape = None
        elif isinstance(output_shape, (tuple, list)):
            self._output_shape = tuple(output_shape)
        else:
            if not callable(output_shape):
                raise TypeError('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape

    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            # With TensorFlow or CNTK, we can infer the output shape directly:
            if K.backend() in ('tensorflow', 'cntk'):
                if isinstance(input_shape, list):
                    xs = [K.placeholder(shape=shape, sparse=is_sparse)
                          for shape, is_sparse in zip(input_shape, self.__is_sparse)]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape, sparse=self.__is_sparse)
                    x = self.call(x)
                if isinstance(x, list):
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # Otherwise, we default to the input shape.
            warnings.warn('`output_shape` argument not specified for layer {} '
                          'and cannot be automatically inferred '
                          'with the Theano backend. '
                          'Defaulting to output shape `{}` '
                          '(same as input shape). '
                          'If the expected output shape is different, '
                          'specify it via the `output_shape` argument.'
                          .format(self.name, input_shape))
            return input_shape
        elif isinstance(self._output_shape, (tuple, list)):
            if isinstance(input_shape, list):
                num_samples = input_shape[0][0]
            else:
                num_samples = input_shape[0] if input_shape else None
            return (num_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError('`output_shape` function must return a tuple or '
                                 'a list of tuples.')
            if isinstance(shape, list):
                if isinstance(shape[0], int) or shape[0] is None:
                    shape = tuple(shape)
            return shape

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, list):
            self.__is_sparse = [K.is_sparse(inp) for inp in inputs]
        else:
            self.__is_sparse = K.is_sparse(inputs)
        return super(Lambda, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None):
        arguments = self.arguments
        if has_arg(self.function, 'mask'):
            arguments['mask'] = mask
        return self.function(inputs, **arguments)

    def compute_mask(self, inputs, mask=None):
        if callable(self.mask):
            return self.mask(inputs, mask)
        return self.mask

    def get_config(self):
        if isinstance(self.function, python_types.LambdaType):
            function = func_dump(self.function)
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            output_shape = func_dump(self._output_shape)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        globs = globals()
        if custom_objects:
            globs = dict(list(globs.items()) + list(custom_objects.items()))
        function_type = config.pop('function_type')
        if function_type == 'function':
            # Simple lookup in custom objects
            function = deserialize_keras_object(
                config['function'],
                custom_objects=custom_objects,
                printable_module_name='function in Lambda layer')
        elif function_type == 'lambda':
            # Unsafe deserialization from bytecode
            function = func_load(config['function'], globs=globs)
        else:
            raise TypeError('Unknown function type:', function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            # Simple lookup in custom objects
            output_shape = deserialize_keras_object(
                config['output_shape'],
                custom_objects=custom_objects,
                printable_module_name='output_shape function in Lambda layer')
        elif output_shape_type == 'lambda':
            # Unsafe deserialization from bytecode
            output_shape = func_load(config['output_shape'], globs=globs)
        else:
            output_shape = config['output_shape']

        # If arguments were numpy array, they have been saved as
        # list. We need to recover the ndarray
        if 'arguments' in config:
            for key in config['arguments']:
                if isinstance(config['arguments'][key], dict):
                    arg_dict = config['arguments'][key]
                    if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
                        # Overwrite the argument with its numpy translation
                        config['arguments'][key] = np.array(arg_dict['value'])

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)


"""Layers that can merge several inputs into one.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K
import tensorflow as tf


# completely copied from keras.merge._Merge class (private).
class _Merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        # Arguments
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        # Returns
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        # Raises
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Can not merge tensors with different '
                             'batch sizes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = K.expand_dims(x, 1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = K.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate(
                            [x_shape[1:], K.expand_dims(batch_size)])
                        x_transposed = K.reshape(x, K.stack(
                            [batch_size, K.prod(x_shape[1:])]))
                        x_transposed = K.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = K.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(K.permute_dimensions(x, dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed, we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = K.shape(y)
                        y_ndim = K.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate(
                            [K.expand_dims(batch_size), y_shape[:y_ndim - 1]])
                        y = K.reshape(y, (-1, batch_size))
                        y = K.permute_dimensions(y, (1, 0))
                        y = K.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = K.permute_dimensions(y, dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class SparseDot(_Merge):
    """Layer that computes a dot product between samples in two tensors.

    E.g. if applied to a list of two tensors `a` and `b` of shape `(batch_size, n)`,
    the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.

    # Arguments
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, axes, normalize=False, **kwargs):
        super(SparseDot, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError('Invalid type for `axes` - '
                                'should be a list or an int.')
            if len(axes) != 2:
                raise ValueError('Invalid format for `axes` - '
                                 'should contain two elements.')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError('Invalid format for `axes` - '
                                 'list elements should be "int".')
        self.axes = axes
        self.normalize = normalize
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on exactly 2 inputs')
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % K.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=axes[0])
            x2 = K.l2_normalize(x2, axis=axes[1])
        print(x1.shape, x2.shape)
        # output = K.batch_dot(x1, x2, axes)
        output = tf.sparse.matmul(x1, x2)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes,
            'normalize': self.normalize,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



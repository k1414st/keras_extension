from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.client import device_lib
from tensorflow.core.protobuf import config_pb2

from collections import defaultdict

import numpy as np
# from distutils.version import StrictVersion
import os

from keras.backend.common import floatx
from keras.backend.common import epsilon
from keras.backend.common import normalize_data_format
from keras.utils.generic_utils import has_arg

# Legacy functions
from .common import set_image_dim_ordering
from .common import image_dim_ordering

py_all = all
py_any = any
py_sum = sum
py_slice = slice

# INTERNAL UTILS

# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None

# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}

# This dictionary holds a mapping {graph: UID_DICT}.
# each UID_DICT is a dictionary mapping name prefixes to a current index,
# used for generating graph-specific string UIDs
# for various names (e.g. layer names).
_GRAPH_UID_DICTS = {}

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change during our lifetime.
_LOCAL_DEVICES = None

name_scope = tf.name_scope


def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    """
    return x.dtype.base_dtype.name


def bias_add(x, bias, data_format=None):
    """Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    """
    data_format = normalize_data_format(data_format)
    bias_shape = int_shape(bias)
    if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
        raise ValueError('Unexpected bias dimensions %d, expect to be 1 or %d dimensions'
                         % (len(bias_shape), ndim(x)))
    if ndim(x) == 5:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                if _has_nchw_support():
                    x = tf.nn.bias_add(x, bias,
                                       data_format='NCHW')
                else:
                    x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x = tf.nn.bias_add(x, bias,
                                   data_format='NHWC')
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1, ) + bias_shape)
    else:
        x = tf.nn.bias_add(x, bias)
    return x


# RANDOMNESS



def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A padded 4D tensor.
    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               [0, 0]]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2))
    return tf.pad(x, pattern)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """Pads 5D tensor with zeros along the depth, height, width dimensions.
    Pads these dimensions with respectively
    "padding[0]", "padding[1]" and "padding[2]" zeros left and right.
    For 'channels_last' data_format,
    the 2nd, 3rd and 4th dimension will be padded.
    For 'channels_first' data_format,
    the 3rd, 4th and 5th dimension will be padded.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A padded 5D tensor.
    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [
        [0, 0],
        [padding[0][0], padding[0][1]],
        [padding[1][0], padding[1][1]],
        [padding[2][0], padding[2][1]],
        [0, 0]
    ]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2, 3))

    return tf.pad(x, pattern)


# CONVOLUTIONS


def _preprocess_conv1d_input(x, data_format):
    """Transpose and cast the input before the conv1d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NWC'  # to pass TF Conv2dNative operations
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 1))  # NCW -> NWC
        else:
            tf_data_format = 'NCW'
    return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
    """Transpose and cast the input before the conv2d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
        force_transpose: boolean, whether force to transpose input from NCHW to NHWC
                        if the `data_format` is `"channels_first"`.
    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support() or force_transpose:
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = 'NCHW'
    return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
    """Transpose and cast the input before the conv3d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NDHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 3, 4, 1))
        else:
            tf_data_format = 'NCDHW'
    return x, tf_data_format


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.
    # Arguments
        padding: string, `"same"` or `"valid"`.
    # Returns
        a string, `"SAME"` or `"VALID"`.
    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding


def conv1d(x, kernel, strides=1, padding='valid',
           data_format=None, dilation_rate=1):
    """1D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilate rate.
    # Returns
        A tensor, result of 1D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    kernel_shape = kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    padding = _preprocess_padding(padding)
    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=(dilation_rate,),
        strides=(strides,),
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NWC':
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW
    return x


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.
    # Returns
        A tensor, result of 2D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)

    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1, 1)):
    """3D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.
    # Returns
        A tensor, result of 3D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)
    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


def get_part_weights(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """partial 2D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.
    # Returns
        A tensor, result of 2D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    print('-'*10)
    print(x.shape)
    print('-'*10)
    print(data_format)
    print('-'*10)
    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    print('-'*10)
    print(x.shape)
    print('-'*10)
    print(tf_data_format)
    print('-'*10)

    if tf_data_format == 'NHWC':
        m1 = tf.ones(shape=(x.shape[1], x.shape[2]))
    elif tf_data_format == 'NCHW':
        m1 = tf.ones(shape=(x.shape[2], x.shape[3]))
    padding_partial = [[kernel[0]//2, kernel[0]//2], [kernel[1]//2, kernel[1]//2]]
    print(padding_partial)
    m1_p0 = tf.pad(m1, tf.constant(padding_partial), 'CONSTANT', constant_values=0)
    # m1_p1 = tf.pad(m1, tf.constant(padding_partial), 'CONSTANT', constant_values=1)

    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x

def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """partial 2D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.
    # Returns
        A tensor, result of 2D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)

    if tf_data_format == 'NHWC':
        m1 = tf.ones(shape=(x.shape[1], x.shape[2]))
    elif tf_data_format == 'NCHW':
        m1 = tf.ones(shape=(x.shape[2], x.shape[3]))
    padding_partial = [[kernel[0]//2, kernel[0]//2], [kernel[1]//2, kernel[1]//2]]
    print(padding_partial)
    # m1_p0 = tf.pad(m1, tf.constant(padding_partial), 'CONSTANT', constant_values=0)
    # m1_p1 = tf.pad(m1, tf.constant(padding_partial), 'CONSTANT', constant_values=1)


    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


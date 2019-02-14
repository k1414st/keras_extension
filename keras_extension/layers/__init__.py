from __future__ import absolute_import
from __future__ import print_function


from .partial_convolutional import *
from .mac import MACCell, MAC
from .graph import GraphConv, GraphRNN

__version__ = '0.0.1'


################################################################################
# import SparseDot if backend is tensorflow.

import os, json

# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if 'KERAS_HOME' in os.environ:
    _keras_dir = os.environ.get('KERAS_HOME')
else:
    _keras_base_dir = os.path.expanduser('~')
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = '/tmp'
    _keras_dir = os.path.join(_keras_base_dir, '.keras')

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _BACKEND)
    _BACKEND = _backend

# Set backend based on KERAS_BACKEND flag, if applicable.
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'tensorflow':
    from .merge import SparseDot

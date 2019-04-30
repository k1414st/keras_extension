# -*- coding: utf-8 -*-
"""
Gaussian Error Linear Unit (GELU)
This module is implementation of "GAUSSIAN ERROR LINEAR UNITS (GELUS)"
(https://arxiv.org/abs/1606.08415)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras.backend as K


def gelu(x, approximate='tanh'):
    """Gaussian Error Linear Unit.
    (implementation of https://arxiv.org/abs/1606.08415)

    # Arguments
        x: Input tensor.
        approximate: Approximation method of GELU.
            "tanh" or "sigmoid" (defaults to "tanh".)

    # Returns
        The gaussian error unit activation: x * Phi(x)
            where Phi(x) = P(X <= x), X ~ Normal(0, 1)
            (approximated by "tanh" or "sigmoid")
    """
    if not approximate in ('tanh', 'sigmoid'):
        raise ValueError('approximate must be in ("tanh", "sigmoid")')
    if approximate == 'tanh':
        return 0.5*x * (1 + K.tanh(np.sqrt(2/np.pi) * \
                                   (x + 0.044715*K.pow(x, 3))))
    else:
        return x * K.sigmoid(1.702*x)



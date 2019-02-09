# -*- coding: utf-8 -*-
"""
Transformation function from "graph adjacency matrix" to
    1. graph laplacian matrix
    2. normalize graph matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse.csr import csr_matrix


# to avoid zero division
epsilon = 1e-7


def _batch_dot(x, y):
    """ execute dot operation for each unit """
    return np.einsum('ijk,ikl->ijl', x, y)


def to_laplacian(mtx,
                 binarize=False,
                 normalize=False,
                 matrix_type=np.array):
    """
    calculate laplacian matrix.

    Args:
        mtx: input matrix
            (2D square matrix or 3D batched square matrix)
        binarize: binarize weighted matrix.
            (if element != 0, binarize to 1)
        normalize: normalize adjacency matrix.
            (A -> D^1/2 `dot` A `dot` D^1/2)
        matrix_type:
            output matrix type (np.array or scipy.sparse.csr.csr_matrix)
            if input_dim == 3 and specified matrix_type == csr_matrix,
            returns np.array(csr_matrix)
    """

    # validation
    if not mtx.ndim in (2, 3):
        raise ValueError('ndim of input matrix must be 2 or 3.')
    if not mtx.shape[-2] == mtx.shape[-1]:
        raise ValueError('input matrix shape must be squared')
    if not matrix_type in (np.array, csr_matrix):
        raise ValueError(
            'matrix type must be "numpy.array" or "scipy.sparse.csr.csr_matrix"')

    if binarize:
        mtx = np.where(mtx == 0, 0., 1.)

    # normalize adjacency matrix. (A -> D^1/2 `dot` A `dot` D^1/2)
    if normalize:
        if mtx.ndim == 2:
            D = np.diag(mtx.sum(axis=-1))
            mtx = np.sqrt(D).dot(mtx.dot(np.sqrt(D)))
        elif mtx.ndim == 3:
            D = np.array([np.diag(m.sum(axis=1)) for m in mtx])
            mtx = _batch_dot(np.sqrt(D), _batch_dot(mtx, np.sqrt(D)))

    if mtx.ndim == 2:
        I = np.eye(mtx.shape[-1])
    elif mtx.ndim == 3:
        I = np.expand_dims(np.eye(mtx.shape[-1]), axis=0)
    mtx_laplacian = I - mtx

    # batch & sparse -> np.array of csr_matrix
    if mtx.ndim == 3 and matrix_type == csr_matrix:
        return np.array([matrix_type(m) for m in mtx_laplacian])
    # np.array or single csr_matrix
    else:
        return matrix_type(mtx_laplacian)


def normalize_graph_matrix(mtx,
                           binarize=False,
                           add_self=False,
                           normalize=False,
                           normalize_input=False,
                           normalize_output=False,
                           matrix_type=np.array):
    """
    Normalize graph matrix or list of matrix.
    normalize operation include binarize, add_self_loop, whole normalization,
    or input/output normalization. (all optional)

    Args:
        mtx: input adjacency matrix (no self loop, weighted or no-weighted).
            (2D square matrix or 3D batched square matrix)
        binarize: binarize weighted matrix.
            (if element != 0, binarize to 1)
        add_self: add Identify matrix (self loop) before normalize.
        normalize: normalize self-adjacency matrix.
            (A -> D^1/2 `dot` A `dot` D^1/2)
        normalize_input: normalize graph input
        normalize_output: normalize graph output
        matrix_type:
            output matrix type (np.array or scipy.sparse.csr.csr_matrix)
            if input_dim == 3 and specified matrix_type == csr_matrix,
            returns np.array(csr_matrix)
    """

    # validation
    if not mtx.ndim in (2, 3):
        raise ValueError('ndim of input matrix must be 2 or 3.')
    if not mtx.shape[-2] == mtx.shape[-1]:
        raise ValueError('input matrix shape must be squared.')
    if not matrix_type in (np.array, csr_matrix):
        raise ValueError(
            'matrix type must be "numpy.array" or "scipy.sparse.csr.csr_matrix".')
    if normalize + normalize_input + normalize_output > 1:
        raise ValueError('multiple normalize options cannt be selected.')

    # fundamental preprocess
    if binarize:
        mtx = np.where(mtx == 0, 0., 1.)
    if add_self:
        mtx += np.eye(mtx.shape[-1])

    # normalize adjacency matrix. (A -> D^1/2 `dot` A `dot` D^1/2)
    if normalize:
        if mtx.ndim == 2:
            D = np.diag(mtx.sum(axis=-1))
            mtx = np.sqrt(D).dot(mtx.dot(np.sqrt(D)))
        elif mtx.ndim == 3:
            D = np.array([np.diag(m.sum(axis=1)) for m in mtx])
            mtx = _batch_dot(np.sqrt(D), _batch_dot(mtx, np.sqrt(D)))

    elif normalize_input:
        D = mtx.sum(axis=-1)
        D = np.where(D>epsilon, D, epsilon)
        mtx = np.einsum('ijk,ij->ijk', mtx, 1/D)
    elif normalize_output:
        D = mtx.sum(axis=-2, keepdims=True)
        D = np.where(D>epsilon, D, epsilon)
        mtx = np.einsum('ijk,ik->ijk', mtx, 1/D)

    # batch & sparse -> np.array of csr_matrix
    if mtx.ndim == 3 and matrix_type == csr_matrix:
        return np.array([matrix_type(m) for m in mtx])
    # np.array or single csr_matrix
    else:
        return matrix_type(mtx)


# -*- coding: utf-8 -*-
"""
Transformation function from "graph adjacency matrix" to
    1. graph laplacian matrix
    2. normalized graph laplacian matrix
    3. graph adjacency matrix (original with option)
    4. normalized adjacency matrix
"""

import numpy as np
from scipy.sparse.csr import csr_matrix

from keras_extension.preprocess import to_laplacian, normalize_graph_matrix


x = np.array([[0, 2], [3, 0]])
xx = to_laplacian(x, binarize=True, matrix_type=csr_matrix)

x = np.array([[[0, 2], [0, 0]]] * 2)
xx = to_laplacian(x, binarize=True, matrix_type=csr_matrix)
print(xx)


x = np.array([[0, 2], [3, 0]])
xx = to_laplacian(x, binarize=True, normalize=True, matrix_type=csr_matrix)

x = np.array([[[0, 2], [3, 0]]] * 2)
x = np.random.binomial(n=1, p=0.4, size=(10, 5, 5))
x = np.array([xx - np.diag(np.diag(xx)) for xx in x])
print(x)
xx = to_laplacian(x, binarize=True, normalize=True, matrix_type=csr_matrix)
print(x)
xx = normalize_graph_matrix(
    x, add_self=False, binarize=False, normalize_output=True, matrix_type=np.array)
print(xx)

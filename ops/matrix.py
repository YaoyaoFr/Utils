import numpy as np
from scipy import stats


def matrix_sort(matrix: np.ndarray,
                top: int = None,
                if_absolute: bool = True,
                if_diagonal: bool = True):
    shape = np.shape(matrix)
    assert len(shape) == 2, 'Matrix must be two dimension.'
    assert shape[0] == shape[1], 'Matrix must be a square matrix.'

    if top is None:
        top = int(shape[0] * (shape[0] - 1) / 2)

    matrix_temp = matrix
    if if_absolute:
        matrix_temp = np.abs(matrix)

    if not if_diagonal:
        matrix_temp = matrix_temp - np.diag(np.diagonal(matrix_temp))

    order = np.zeros(shape=shape)
    weights = np.zeros(shape=shape)
    edges = []
    for i in range(top):
        index = np.argmax(matrix_temp)
        m, n = divmod(index, shape[0])
        weights[m, n] = matrix[m, n]
        matrix_temp[m, n] = 0
        matrix_temp[n, m] = 0
        order[m, n] = i
        edges.append([int(m), int(n)])
    edges = np.array(edges)

    return order, edges, weights


def vector_sort(vector: np.ndarray,
                top: int = None,
                if_absolute: bool = True,
                ):
    shape = np.shape(vector)
    assert len(shape) == 1, 'Matrix must be two dimension.'

    if top is None:
        top = shape[0]

    if if_absolute:
        vector_temp = np.abs(vector)

    order = np.zeros(shape=shape)
    elements = np.zeros(shape=shape)
    edges = []
    for i in range(top):
        index = np.argmax(vector_temp)
        elements[index] = vector[index]
        vector_temp[index] = 0
        order[index] = i
        edges.append(index)
    edges = np.array(edges)

    return order, edges, elements


def matrix_significance_difference(matrix1, matrix2):
    shape1 = np.shape(matrix1)
    shape2 = np.shape(matrix2)

    assert shape1[1:] == shape2[1:], 'Two matrices must have same shape.'

    significance = np.ones(shape=shape1[1:])

    for i in range(shape1[1]):
        for j in range(i):
            result = stats.levene(matrix1[:, i, j], matrix2[:, i, j])
            significance[i, j] = result.pvalue

    return significance

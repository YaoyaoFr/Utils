'''
Author: your name
Date: 2019-12-23 09:22:44
LastEditTime: 2020-09-02 17:17:54
LastEditors: Yaoyao
Description: In User Settings Edit
FilePath: /DeepLearning2.0/home/ai/data/yaoyao/Program/Python/Utils/ops/matrix.py
'''
import collections
import numpy as np
from scipy import stats
from Dataset.utils.basic import onehot2vector


def matrix_sort(matrix: np.ndarray,
                top: int = None,
                if_absolute: bool = True,
                if_diagonal: bool = True,
                if_symmetric: bool = None,
                ):
    shape = np.shape(matrix)
    assert len(shape) == 2, 'Matrix must be two dimension.'
    assert shape[0] == shape[1], 'Matrix must be a square matrix.'

    # Whether the matrix is symmetric
    if if_symmetric is None:
        if np.max(matrix.T - matrix) == 0:
            if_symmetric = True
        else:
            if_symmetric = False

    # The number of top values
    if top is None:
        top = np.prod(shape)

    if if_absolute:
        matrix_temp = np.abs(matrix)
    else:
        matrix_temp = np.array(matrix)

    if not if_diagonal:
        matrix_temp[np.diag_indices(shape[0])] = 0

    min_value = np.min(matrix_temp) - 1

    # The results are listed in a ordered collection from top to down.
    results = collections.OrderedDict()
    for index in range(top):
        element = {}
        top_index = np.argmax(matrix_temp)
        m, n = divmod(top_index, shape[0])

        element['coordinate'] = [m, n]
        element['value'] = matrix[m, n]

        matrix_temp[m, n] = min_value
        if if_symmetric:
            matrix_temp[n, m] = min_value

        results[index] = element

    return results


def vector_sort(vector: np.ndarray,
                top: int = None,
                if_absolute: bool = True,
                ):
    shape = np.shape(vector)
    assert len(
        shape) == 1, 'Vector must be one dimension but got {:}.'.format(shape)

    if top is None:
        top = shape[0]

    if if_absolute:
        vector_temp = np.abs(vector)
    else:
        vector_temp = np.array(vector)

    min_value = np.min(vector_temp - 1)

    # The results are listed in a ordered collection from top to down.
    results = collections.OrderedDict()
    for index in range(top):
        element = {}
        top_index = np.argmax(vector_temp)

        element['coordinate'] = top_index
        element['value'] = vector[top_index]

        vector_temp[top_index] = min_value

        results[index] = element

    return results


def matrix_significance_difference(
    matrix1: np.ndarray = None,
    matrix2: np.ndarray = None,
    matrix: np.ndarray = None,
    label: np.ndarray = None,
    if_t_test: bool = True,
    threshold: float = None,
    if_mean_values: bool = True,
):
    assert (matrix1 is not None and matrix2 is not None) or (
        matrix is not None and label is not None), 'Dataset doesn\'t complete.'

    if (matrix1 is None or matrix2 is None):
        assert len(matrix) == len(
            label), 'The sample size doesn\'t match, got {:} and {:}.'.format(len(matrix), len(label))

        if len(np.shape(label)) == 2:
            label = onehot2vector(label, 2)

        assert len(np.shape(label)) == 1, 'The label should be a vector but got shape {:}.'.format(
            np.shape(label))

        matrix1 = matrix[np.where(label == 0)]
        matrix2 = matrix[np.where(label == 1)]

    shape1 = np.shape(matrix1)
    shape2 = np.shape(matrix2)

    assert shape1[1:] == shape2[1:], 'Two matrices must have same shape.'

    raw_shape = shape1[1:]
    feature_dim = np.prod(shape1[1:])
    matrix1 = np.reshape(matrix1, newshape=[-1, feature_dim])
    matrix2 = np.reshape(matrix2, newshape=[-1, feature_dim])

    results = {}
    if if_mean_values:
        results['Mean values 0'] = np.reshape(
            np.mean(matrix1, axis=0), newshape=raw_shape)
        results['Mean values 1'] = np.reshape(
            np.mean(matrix2, axis=0), newshape=raw_shape)

    if if_t_test:
        p_value = np.ones(shape=feature_dim)
        statistic = np.zeros(shape=feature_dim)
        for i in range(feature_dim):
            print(
                '\rStatistical analysis of {:}/{:}...'.format(i+1, feature_dim), end='')
            levene_result = stats.levene(matrix1[:, i], matrix2[:, i])
            equal_var = True if levene_result.pvalue > 0.5 else False

            result = stats.ttest_ind(
                matrix1[:, i], matrix2[:, i], equal_var=equal_var)
            if not np.isnan(result.pvalue):
                p_value[i] = result.pvalue
            if not np.isnan(result.statistic):
                statistic[i] = result.statistic
        print('\nStatistical analysis complete.')

        if threshold is not None:
            significance = np.zeros(shape=feature_dim)
            significance[np.where(p_value < threshold)] = 1
            results['significance'] = np.reshape(
                significance, newshape=raw_shape)

        results['p value'] = np.reshape(p_value, newshape=raw_shape)
        results['statistic'] = np.reshape(statistic, newshape=raw_shape)

    return results


def get_edges(matrix: np.ndarray,
              top: int = None,
              if_symmetric: bool = None,
              output_path: str = None):
    shape = np.shape(matrix)
    assert len(shape) == 2, 'The rank of input matrix must be to but get {:d}.'.format(
        len(shape))
    assert shape[0] == shape[1], 'The input matrix must be a square matrix but get shape {:}'.format(
        shape)

    if top is not None:
        matrix_top = np.zeros(shape=shape)

        sorted_matrix = matrix_sort(
            matrix=matrix, top=top, if_symmetric=if_symmetric)
        for index, element in sorted_matrix.items():
            [m, n] = element['coordinate']
            matrix_top[m, n] = element['value']
    else:
        matrix_top = np.copy(matrix)

    np.savetxt(output_path, matrix_top)
    # print('Save matrix to {:s}.'.format(output_path))

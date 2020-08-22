import numpy as np


def sparse_to_matrix(shape: list,
                     sparse: dict = None,
                     indices: np.ndarray or list = None,
                     value: int or list = 1,
                     dtype: type = None):
    matrix = np.zeros(shape=shape)

    assert sparse is not None or indices is not None, 'Must feed a dict or list!'

    if indices is None:
        for coordinate in sparse:
            matrix[coordinate] = sparse[coordinate]
    elif sparse is None:
        if isinstance(indices, np.ndarray):
            indices = indices.astype(int)
            for index in range(np.size(indices, 0)):
                coordinate = tuple(indices[index, :])
                v = value if isinstance(value, int) else value[index]
                try:
                    matrix[coordinate] = v
                except:
                    pass

    matrix = matrix.astype(dtype=dtype)
    return matrix


def matrix_to_sparse(data: np.ndarray):
    """
    Transfer a matrix into a sparse format
    :param data: Input data
    :return:
    """

    sparse_matrix = {}

    shape = np.shape(data)
    if len(shape) >= 1:
        for x in range(shape[0]):
            if len(shape) >= 2:
                for y in range(shape[1]):
                    if len(shape) >= 3:
                        for z in range(shape[2]):
                            if len(shape) >= 4:
                                raise TypeError('expected rank <= 3 dense array or matrix')
                            else:
                                if data[x, y, z] != 0:
                                    sparse_matrix[(x, y, z)] = data[x, y, z]
                    else:
                        if data[x, y] != 0:
                            sparse_matrix[(x, y)] = data[x, y]
            else:
                if data[x] != 0:
                    sparse_matrix[(x)] = data[x]
    else:
        raise TypeError('expected rank >=1 dense array or matrix')

    shape = np.array(np.shape(data), dtype=np.int64)
    indices = np.array([list(item[0]) if not isinstance(item[0], int) else item[0]
                        for item in sparse_matrix.items()], dtype=np.int64)
    values = np.array([item[1] for item in sparse_matrix.items()])

    return {'sparse_matrix': sparse_matrix,
            'shape': shape,
            'indices': indices,
            'values': values,
            }


def sparse_mask(a: dict,
                b: dict):
    indices_a = a['indices']
    indices_b = b['indices']

    assert np.size(indices_a, 0) >= np.size(indices_b, 0), \
        'The sample size of A must greater than B. '

    assert np.size(indices_a, 1) == np.size(indices_b, 1), \
        'The rank of A and B must be same but go {:} and {:}'.format(a['indices'], b['indices'])

    indices_b = [list(indices_b[index])
                 for index in range(np.size(indices_b, 0))]
    mask_indices = []
    indices = []
    for index, indice in enumerate(indices_a):
        if list(indice) in indices_b:
            indices.append(list(indice))
        else:
            mask_indices.append(index)
    return mask_indices, indices

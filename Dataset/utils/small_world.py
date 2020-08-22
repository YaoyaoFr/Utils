'''
Author: your name
Date: 2020-06-11 08:54:58
LastEditTime: 2020-08-22 11:06:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Utils/Dataset/utils/small_world.py
'''
import numpy as np
import h5py
import sklearn.preprocessing as prep


def local_connectivity_pattern_extraction_fold(data_fold: dict, threshold: float = 0):
    for p in ['train', 'valid', 'test']:
        if '{:s} data'.format(p) not in data_fold:
            continue

        functional_connectivity = data_fold['{:s} data'.format(p)]

        absolute_FC = np.abs(functional_connectivity)
        # Thresholding the insignificant connectivities
        functional_connectivity[np.where(
            np.abs(absolute_FC) < threshold)] = 0
        mask_FC = np.cast[int](absolute_FC >= threshold)

        [sample_size, node_num, node_num,
         channel_num] = np.shape(functional_connectivity)
        node_rows = np.split(absolute_FC, indices_or_sections=node_num, axis=1)
        mask_rows = np.split(mask_FC, indices_or_sections=node_num, axis=1)

        connectivity_patterns = []
        mask_connectivity_patterns = []
        for node_row, mask_row in zip(node_rows, mask_rows):
        # for node_row in node_rows:
            plus = node_row + np.transpose(node_row, axes=[0, 2, 1, 3])
            mask = mask_row * np.transpose(mask_row, axes=[0, 2, 1, 3])
            connectivity_patterns.append(plus)
            mask_connectivity_patterns.append(plus * mask)
        connectivity_patterns = np.concatenate(connectivity_patterns, axis=-1)
        mask_connectivity_patterns = np.concatenate(
            mask_connectivity_patterns, axis=-1)

        data_fold['{:s} CPs'.format(p)] = connectivity_patterns
        data_fold['{:s} maskCPs'.format(p)] = mask_connectivity_patterns

    return data_fold


def data_normalization_fold_SM(
    data_fold: h5py.Group or dict,
    fit_data: np.ndarray = None,
    strategy: str = 'standarization',
    tag: str = 'CPs',
    node_num: int = 90,
):

    if isinstance(data_fold, h5py.Group):
        data_fold = {name: np.array(data_fold[name]) for name in data_fold}

    if not fit_data:
        fit_datas = []
        for node_index in range(node_num):
            fit_data = []
            for index in data_fold:
                if tag not in index:
                    continue

                fit_data.append(data_fold[index][..., node_index])

            fit_data = np.concatenate(fit_data, axis=0)
            shape = np.shape(fit_data)
            feature_dim = np.prod(shape[1:])
            fit_data = np.reshape(fit_data, newshape=[-1, feature_dim])
            fit_datas.append(fit_data)
    else:
        shape = np.shape(fit_data)
        feature_dim = np.prod(shape[1:])
        fit_data = np.reshape(fit_data, newshape=[-1, feature_dim])

    preprocessors = [prep.StandardScaler().fit(fit_data)
                     for fit_data in fit_datas]
    for index in data_fold:
        if tag not in index:
            continue

        # Preprocessing
        datas = []
        for node_index, preprocessor in zip(range(node_num), preprocessors):
            data = data_fold[index][..., node_index]
            shape = np.shape(data)
            data = np.reshape(data, [-1, feature_dim])

            # Processing
            if strategy == 'standarization':
                data = preprocessor.transform(data)
            else:
                raise TypeError(
                    'The normalization strategy supported must in [\'standarization\']')

            # Postprocessing
            data = np.reshape(data, newshape=shape)
            if len(shape) < 4:
                data = np.expand_dims(data, axis=-1)
            datas.append(data)
        data_fold[index] = np.concatenate(datas, axis=-1)
    return data_fold

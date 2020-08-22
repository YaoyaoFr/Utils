import contextlib
import math
import multiprocessing
import os
import string
import sys
import time

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE

from ops import graph as graph
from ops.sparse import matrix_to_sparse, sparse_to_matrix


def hdf5_handler(filename, mode="r"):
    while True:
        try:
            h5py.File(filename, "a").close()
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            settings = list(propfaid.get_cache())
            settings[1] = 0
            settings[2] = 0
            propfaid.set_cache(*settings)
            with contextlib.closing(h5py.h5f.open(filename,
                                                  fapl=propfaid)) as fid:
                return h5py.File(fid, mode)
        except Exception:
            print('Loading file {:} error, pause 3 seconds and reexcution.'.
                  format(filename))
            time.sleep(3)
            continue


def create_dataset_hdf5(group: h5py.Group,
                        data: np.ndarray,
                        name: str,
                        is_save: bool = True,
                        cover: bool = True,
                        dtype: np.dtype = float,
                        show_info: bool = True) -> None:
    try:
        if name in group:
            if cover:
                group.pop(name)
            else:
                if show_info:
                    print('\'{:s}\' already exist in \'{:s}\'.'.format(
                        name, group.name))
                return
    except Exception:
        print('Creat dataset {:s} into {:s} error'.format(name, group.name))

    if is_save:
        group.create_dataset(name=name, data=data, dtype=dtype)
        if show_info:
            print('Create \'{:s}\' in \'{:s}\''.format(name, group.name))


def prepare_classify_data(folds: h5py.Group = None,
                          new_shape: list = None,
                          one_hot: bool = False,
                          normalization: bool = False,
                          data_flag: str = 'data encoder',
                          slice_index: int or list = None):
    if folds is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')
        folds = hdf5.require_group('scheme 1/falff')

    data_list = list()
    for fold_idx in folds:
        # Load Data
        fold = folds[fold_idx]
        try:
            data = dict()
            for tvt in ['train', 'valid', 'test']:
                tvt_data = '{:s} {:s}'.format(tvt, data_flag)
                data_tmp = np.array(fold[tvt_data])

                if slice_index is not None:
                    data_tmp = data_tmp[:, slice_index, :]
                if new_shape is not None:
                    data_tmp = np.reshape(a=data_tmp, newshape=new_shape)
                if normalization:
                    data_tmp, mean, std = data_normalization(
                        data=data_tmp,
                        axis=1,
                        standardization=True,
                        sigmoid=False)

                tvt_label = '{:s} label'.format(tvt)
                label_tmp = np.array(fold[tvt_label])

                if one_hot:
                    label_tmp = vector2onehot(label_tmp, 2)

                data['{:s} data'.format(tvt)] = data_tmp
                data[tvt_label] = label_tmp
            data_list.append(data)
            print('{:s} prepared.'.format(fold.name))
        except:
            print('{:s} prepared failed.'.format(fold.name))

    return data_list


def compute_connectivity(functional):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional))
        corr[np.eye(N=np.size(corr, 0)) == 1] = 1
        return corr
        # mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        # m = ma.masked_where(mask == 1, mask)
        # return ma.masked_where(m, corr).compressed()


def set_diagonal_to_zero(data: np.ndarray):
    [sample_size, node_num, node_num] = np.shape(data)
    for index_sample in range(sample_size):
        for index_node in range(node_num):
            data[index_sample, index_node, index_node] = 0

    return data


class SafeFormat(dict):
    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(dd))


def run_progress(callable_func, items, message=None, jobs=10):
    results = []

    print('Starting pool of %d jobs' % jobs)

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    # Or allocate a pool for multi-threading
    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func,
                             args=(item, ),
                             callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    print
    return results


def split_slices(data, axis=[0, 3, 1, 2, 4]):
    """
    Split data to slices
    :param data: with the shape of [batch_num, width, height, depth, channels]
    :param axis: one of the (0, 1, 2) corresponding (width, height, depth)
    :return:
    """

    data = data.transpose(axis)
    shape = np.shape(data)
    data = np.reshape(
        a=data,
        newshape=[shape[0] * shape[1], shape[2], shape[3], shape[4]],
    )
    return data


def data_normalization(data: np.ndarray or list,
                       mean: np.ndarray = None,
                       std: np.ndarray = None,
                       axis: int = 0,
                       normalization: bool = True,
                       standardization: bool = False,
                       sigmoid: bool = False):
    # Format list to ndarray
    if isinstance(data, list):
        data = np.array(data)

    if normalization:
        if mean is None:
            mean = np.mean(data, axis=axis)
        if std is None:
            std = np.std(data, axis=axis)
            std[std == 0] = 1
        data = (data - mean) / std

    if standardization:
        max_v = np.max(data, axis=axis)
        min_v = np.min(data, axis=axis)
        scale = max_v - min_v
        scale[scale == 0] = 1
        data = (data - min_v) / scale

    if sigmoid:
        data = 1.0 / (1 + np.exp(-data)) - 0.5
    return data, mean, std


def data_normalization(data: np.ndarray):
    shape = np.shape(data)
    data = np.reshape(data, [shape[0], -1])
    preprocessor = prep.StandardScaler().fit(data)
    data = preprocessor.transform(data)
    data = np.reshape(data, newshape=shape)
    return data


def data_normalization_fold(
    data_fold: h5py.Group or dict,
    fit_data: np.ndarray = None,
    strategy: str = 'standarization',
    tag: str = 'data',
):
    if isinstance(data_fold, h5py.Group):
        data_fold = {name: np.array(data_fold[name]) for name in data_fold}

    if not fit_data:
        fit_data = []
        for index in data_fold:
            if tag not in index:
                continue

            fit_data.append(data_fold[index])

        fit_data = np.concatenate(fit_data, axis=0)
        shape = np.shape(fit_data)
        feature_dim = np.prod(shape[1:])
        fit_data = np.reshape(fit_data, newshape=[-1, feature_dim])
    else:
        shape = np.shape(fit_data)
        feature_dim = np.prod(shape[1:])
        fit_data = np.reshape(fit_data, newshape=[-1, feature_dim])

    preprocessor = prep.StandardScaler().fit(fit_data)
    for index in data_fold:
        if tag not in index:
            continue

        # Preprocessing
        data = data_fold[index]
        shape = np.shape(data)
        data = np.reshape(data, [-1, feature_dim])

        # Processing
        if strategy == 'standarization':
            data = preprocessor.transform(data)
        elif strategy == 'normalization':
            data_max = np.max(data, axis=0, keepdims=True)
            data_min = np.min(data, axis=0, keepdims=True)
            data = (data - data_min) / (data_max - data_min)
            data = np.nan_to_num(data)

        # Postprocessing
        data = np.reshape(data, newshape=shape)
        if len(shape) < 4:
            data = np.expand_dims(data, axis=-1)
        data_fold[index] = data
    return data_fold


def vector2onehot(data, class_num: int = None):
    if class_num is None:
        class_num = np.max(data) + 1

    data_tmp = np.zeros([np.size(data, 0), class_num])
    for class_index in range(class_num):
        data_tmp[np.where(data == class_index)[0], class_index] = 1
    return data_tmp


def get_folds(dataset: str,
              feature: str,
              fold_indexes: list = None) -> h5py.Group:
    hdf5_path = 'F:/OneDriveOffL/Data/Data/{:s}/{:s}.hdf5'.format(
        dataset.upper(), dataset.lower()).encode()
    hdf5 = hdf5_handler(hdf5_path, 'a')
    folds_basic = hdf5['experiments/{:s}_whole'.format(feature)]

    if fold_indexes is None:
        fold_indexes = folds_basic.keys()

    folds = dict()
    for fold_idx in fold_indexes:
        folds[fold_idx] = folds_basic[fold_idx]

    return folds


def get_datas(dataset, feature, fold_indexes, indexes, tvt='train'):
    folds = get_folds(dataset=dataset,
                      feature=feature,
                      fold_indexes=fold_indexes)
    datas = dict()
    for fold_idx in folds:
        fold = folds[fold_idx]
        data_basic = np.array(fold['{:s} data'.format(tvt)])
        recons_basic = np.array(fold['{:s} data output'.format(tvt)])

        data = dict()
        data['data'] = data_basic[indexes, :, :, 0]
        data['output'] = recons_basic[indexes, :, :, 0]
        datas[fold_idx] = data

    return datas


def get_subjects(pheno: pd.DataFrame, group: str) -> pd.Series:
    dx_group = {'health': 1, 'patient': 0, 'all': 2}[group]
    pheno = pheno[pheno['DX_GROUP'] != dx_group]
    ids = pheno['FILE_ID']
    ids_encode = list()
    for id in ids:
        ids_encode.append(id.encode())
    ids_encode = pd.Series(ids_encode)
    return ids_encode


def extract_regions(data: np.ndarray,
                    regions: int or list = None,
                    atlas: h5py.Group = None,
                    mask: bool = False) -> list:
    if atlas is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_aal.hdf5'
        atlas = hdf5_handler(hdf5_path)['MNI']

    if regions is None:
        regions = range(90)
    elif isinstance(regions, int):
        regions = range(regions)

    shape = np.shape(data)
    batch_size = shape[0]
    if len(shape) > 4:
        channel_num = shape[-1]
    else:
        data = np.expand_dims(data, axis=-1)
        channel_num = 1

    data_list = []
    for region_index in regions:
        data_region = []
        for channel in range(channel_num):
            data_channel = data[:, :, :, :, channel]
            region_index_str = str(region_index + 1)
            region_group = atlas[region_index_str]
            bounds = np.array(region_group['bounds'], dtype=int)

            data_channel = data_channel[:, bounds[0][0]:bounds[0][1],
                                        bounds[1][0]:bounds[1][1],
                                        bounds[2][0]:bounds[2][1], ]

            if mask:
                mask = np.concatenate([
                    np.expand_dims(np.array(region_group['mask'], dtype=int),
                                   0) for _ in range(batch_size)
                ],
                                      axis=0)
                data_channel[mask] = 0

            data_region.append(np.expand_dims(data_channel, axis=-1))
        data_region = np.concatenate(data_region, axis=-1)
        data_list.append(data_region)
    return data_list


def t_test(
    normal_controls: np.ndarray,
    patients: np.ndarray,
    significance: float = 0.05,
    mask: dict = None,
) -> dict:
    """
    Independent two-sample t-test of normal controls and patients for each element of the functional feature
    :param normal_controls: The functional feature of normal controls with the shape of [data_size, width, height, (deep)]
    :param patients: The functional feature of patients with the shape of [data_size, width, height, (deep)]
    :param significance: The threshold of statistic significance, default=0.05
    :param mask: The index of element to be test
    :return: A dictionary contains:
                hypothesis: np.ndarray(bool), indicate whether accept the null hypothesis \mu_1=\mu_2
                t_value: np.ndarray(float), the statistic value
                p_value: np.ndarray(float), the calculated p-value
    """

    shape1 = np.shape(normal_controls)
    shape2 = np.shape(patients)
    assert shape1[1:] == shape2[
        1:], 'The shape of normal controls and patients not match!'

    shape = shape1[1:]
    transpose_axes = [i for i in np.arange(start=1, stop=len(shape) + 1)]
    transpose_axes.append(0)
    normal_controls = np.transpose(normal_controls, axes=transpose_axes)
    patients = np.transpose(patients, axes=transpose_axes)

    t_value = np.zeros(shape=shape, dtype=float)
    p_value = np.ones(shape=shape, dtype=float)

    if mask is not None:
        indexes = mask
    else:
        indexes = matrix_to_sparse(np.ones(shape=shape))['indices']

    zero_count = 0
    for index in indexes:
        index = tuple(index)
        nc = normal_controls[index]
        patient = patients[index]

        if not any(np.concatenate((nc, patient))):
            zero_count += 1
            # print('{:d}: {:}'.format(zero_count, index))
            continue

        result = stats.levene(nc, patient)
        equal_var = result.pvalue > significance
        (statistic, pvalue) = stats.ttest_ind(nc, patient, equal_var=equal_var)

        if not math.isnan(statistic):
            t_value[index] = statistic
        if not math.isnan(pvalue):
            p_value[index] = pvalue

    hypothesis = p_value < significance

    return {
        'hypothesis': hypothesis,
        't_value': t_value,
        'p_value': p_value,
    }


# def select_ROIs(feature_group: h5py.Group = None,
#                 dataset: str = 'ABIDE',
#                 feature: str = 'reho',
#                 aal_atlas: np.ndarray = None,
#                 top_k_ROI: int = 5,
#                 ):
#     if feature_group is None:
#         hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
#         feature_group = hdf5_handler(hdf5_path)['{:s}/statistic/{:s}'.format(dataset, feature)]
#
#     if aal_atlas is None:
#         aal_path = 'Data/AAL/aal_61_73_61.nii'
#         aal_atlas = load_nifti_data(aal_path)
#
#     if feature in ['reho', 'falff']:
#         t_value = np.array(feature_group['t_value'])
#         t_value = matrix_to_sparse(t_value)['sparse_matrix']
#         t_value = [item for item in t_value.items()]
#
#         # Calculate the sum of t-value in each ROI
#         t_value_sum = {}
#         for t_tuple in t_value:
#             coordinate, t = t_tuple
#             ROI_index = aal_atlas[coordinate]
#             if ROI_index in t_value_sum:
#                 t_value_sum[ROI_index] += t
#             else:
#                 t_value_sum[ROI_index] = t
#
#         # Divide the sum of t-value by voxel num in each brain region
#         path = b'F:/OneDriveOffL/Data/Data/DCAE_aal.hdf5'
#         aal = hdf5_handler(path)
#         # for ROI_index in t_value_sum:
#         #     t = t_value_sum[ROI_index]
#         #     voxel_num = aal['MNI/{:d}/voxel_num'.format(ROI_index)].value
#         #     t_value_sum[ROI_index] = t / voxel_num
#
#         # Sort the sum of t-value
#         t_value_sum = [item for item in t_value_sum.items()]
#         t_value_sum = sorted(t_value_sum, key=lambda x: x[1], reverse=True)
#         return [ROI[0] for ROI in t_value_sum[:top_k_ROI]]


def select_landmarks(dataset: str, feature: str, landmk_num: int):
    hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
    statistic_group = hdf5_handler(hdf5_path)['{:s}/statistic/{:s}'.format(
        dataset, feature)]
    p_value = np.array(statistic_group['p_value'])
    hypothesis = np.array(statistic_group['hypothesis']).astype(bool)
    p_value[hypothesis] = 0
    p_value_sparse = matrix_to_sparse(p_value)['sparse_matrix']
    p_value = sorted([item for item in p_value_sparse.items()],
                     key=lambda x: x[1])

    # In this code, we attached the pre-trained model which were trained with 40 landmarks
    landmks = np.array([p_tuple[0] for p_tuple in p_value[:landmk_num]])
    return landmks


#
# def matrix_to_sparse(data: np.ndarray):
#     """
#     Transfer a matrix into a sparse format
#     :param data: Input data
#     :return:
#     """
#
#     sparse_matrix = {}
#
#     shape = np.shape(data)
#     if len(shape) >= 1:
#         for x in range(shape[0]):
#             if len(shape) >= 2:
#                 for y in range(shape[1]):
#                     if len(shape) >= 3:
#                         for z in range(shape[2]):
#                             if len(shape) >= 4:
#                                 raise TypeError('expected rank <= 3 dense array or matrix')
#                             else:
#                                 if data[x, y, z] != 0:
#                                     sparse_matrix[(x, y, z)] = data[x, y, z]
#                     else:
#                         if data[x, y] != 0:
#                             sparse_matrix[(x, y)] = data[x, y]
#             else:
#                 if data[x] != 0:
#                     sparse_matrix[x] = data[x]
#     else:
#         raise TypeError('expected rank >=1 dense array or matrix')
#
#     return sparse_matrix


def get_data_hdf5(file_name='data') -> h5py.Group:
    """
    Load the data hdf5 file which has the structure:
    <{dataset}> string, The name of dataset optional in ['ABIDE', 'ABIDE II', 'ADHD', 'FCP'].
        <statistic>
            <{feature}> string,
                "hypothesis"
                "p_value"
                "t_value"
        <subjects>
            "{feature}" np.ndarray, the data of corresponding feature optional in ['FC', 'falff', 'reho', 'vmhc'].
    :param file_name:
    :return:
    """
    hdf5_path = '{:s}/{:s}_{:s}.hdf5'.format(basic_path, project_name,
                                             file_name).encode()
    return hdf5_handler(hdf5_path)


def get_scheme_hdf5(file_name='scheme') -> h5py.Group:
    """
    Load the data hdf5 file which has the structure:
    <{dataset}> string, The name of dataset optional in ['ABIDE', 'ABIDE II', 'ADHD', 'FCP'].
        <statistic>
            <{feature}> string,
                "hypothesis"
                "p_value"
                "t_value"
        <subjects>
            "{feature}" np.ndarray, the data of corresponding feature optional in ['FC', 'falff', 'reho', 'vmhc'].
    :param file_name:
    :return:
    """
    hdf5_path = '{:s}/{:s}_{:s}.hdf5'.format(basic_path, project_name,
                                             file_name).encode()
    return hdf5_handler(hdf5_path)


def get_folds_hdf5(file_name='folds') -> h5py.Group:
    """
    Load the folds hdf5 file which has the structure:
    <{dataset}> string, The name of dataset optional in ['ABIDE', 'ABIDE II', 'ADHD', 'FCP'].
        <{fold_index}> int, The index of folds
            "train": list, The list of subjects within train set.
            "valid": list The list subjects within validate set
            "test": list, The list of subjects within test set.
    :param file_name:
    :return: the folds hdf5 file/group
    """
    hdf5_path = '{:s}/{:s}_{:s}.hdf5'.format(basic_path, project_name,
                                             file_name).encode()
    return hdf5_handler(hdf5_path)


class AAL:
    """
    Load the aal hdf5 file which has the structure:
    <{space}> string, optional in ['MNI']
        <{resolution}> string, a number list joint by '_', optional in ['61_73_61', '181_217_181']
            "atlas": np.ndarray that consist of the label for each brain region
            <metric> string, metric optional in ['Euclidean', 'Adjacent']
                "distance": np.ndarray element in (i, j) indicate the distance between the i-th and j-th
                            brain region with the metric
                <{nearest-k}> string of the number of top-k nearest neighbor in the metric
                    "adj_matrix": np.ndarray, the basic adjacency matrix
                    "{depth}": np.ndarray, the adjacency matrix in the depth
            <{brain_regions}> string, the index of brain regions range from [1, ROI_num]
                "bound": np.ndarray with shape [3, 2] indicate the start and end coordinate
                            in the corresponding space.
                "dimension": np.ndarray with shape [3, 1] indicate the size of minimal cube contains
                                the brain region
                "mask": np.ndarray with shape of "dimension", and element 0 indicate that the corresponding voxel
                            belong to this brain region.
                "voxel_num": int indicate the number of voxel belong to this brain region.


    Note: <>, "", '' indicate group, dataset and attribute respectively,
    :return: hdf5 file contains the information of AAL atlas
    """
    def __init__(self, file_name='aal'):
        hdf5_path = '{:s}/{:s}_{:s}.hdf5'.format(basic_path, project_name,
                                                 file_name).encode()
        self.aal_group = hdf5_handler(hdf5_path)

    def get_aal_hdf5(self) -> h5py.Group:
        return self.aal_group

    def calculate_structure(
        self,
        space: str = 'MNI',
        resolution: str = '61_73_61',
        metric: str = 'Euclidean',
        nearest_k: int = 10,
    ):

        basic_group = self.aal_group['{:s}/{:s}'.format(space, resolution)]
        atlas = np.array(basic_group['atlas'])
        ROI_num = int(np.max(atlas))

        metric_group = basic_group.require_group(metric)
        whole_brain_graph = np.zeros(shape=[ROI_num, ROI_num])
        if metric == 'Euclidean':
            distance = calculate_distance(
                atlas=atlas,
                ROI_num=ROI_num,
                metric=metric,
            )
            for region_index in np.arange(ROI_num):
                # Firstly, we sort the list of distance to other regions for the chosen region
                distance_to_ROI = sorted(matrix_to_sparse(
                    distance[region_index, :])['sparse_matrix'].items(),
                                         key=lambda x: x[1])

                # Secondly, we select the nearest-k edges connected with chosen region,
                # then set the corresponding element in the whole brain graph to 1.
                nearest_k_list = [t[0] for t in distance_to_ROI[:nearest_k]]
                nearest_k_to_ROI = np.zeros(shape=[ROI_num, 1])
                nearest_k_to_ROI[nearest_k_list] = 1
                whole_brain_graph[:,
                                  region_index] = np.squeeze(nearest_k_to_ROI)
                whole_brain_graph[region_index, :] = np.squeeze(
                    nearest_k_to_ROI)

            create_dataset_hdf5(group=metric_group,
                                name='distance',
                                data=distance)
            create_dataset_hdf5(group=metric_group.require_group(
                '{:d}'.format(nearest_k)),
                                name='adj_matrix',
                                data=whole_brain_graph)
        elif metric == 'Adjacent':
            for i in range(ROI_num):
                ROI_atlas = np.zeros(shape=np.shape(atlas))
                ROI_atlas[atlas == i + 1] = 1
                indices = matrix_to_sparse(ROI_atlas)['indices']
                shape = np.shape(indices)

                indices_expand = [indices]
                for dim in range(shape[1]):
                    expand_matrix1 = np.zeros(shape=shape)
                    expand_matrix1[:, dim] = np.ones(shape=[
                        shape[0],
                    ])
                    expand_matrix2 = -expand_matrix1
                    indices_expand.append(indices + expand_matrix1)
                    indices_expand.append(indices + expand_matrix2)
                indices_expand = np.concatenate(indices_expand, axis=0)

                ROI_atlas_expand = sparse_to_matrix(indices=indices_expand,
                                                    shape=np.shape(atlas))

                for j in range(i + 1, ROI_num):
                    ROI_atlas_j = np.zeros(shape=np.shape(atlas))
                    ROI_atlas_j[atlas == j + 1] = 1
                    if np.sum(ROI_atlas_expand * ROI_atlas_j) > 0:
                        whole_brain_graph[i, j] = 1
                        whole_brain_graph[j, i] = 1
            create_dataset_hdf5(group=metric_group,
                                name='adj_matrix',
                                data=whole_brain_graph)
        return whole_brain_graph

    def get_adj_matrix(
        self,
        nearest_k: int = None,
        depth: int = 0,
        sparse: str = 'dense',
        space: str = 'MNI',
        resolution: str = '61_73_61',
        metric: str = 'Euclidean',
        ROI_num: int = 90,
    ):
        if metric == 'Euclidean':
            if depth == 0:
                adj_matrix = np.array(
                    self.aal_group['{:s}/{:s}/{:s}/{:d}/adj_matrix'.format(
                        space, resolution, metric, nearest_k)])
            else:
                adj_matrix = np.array(
                    self.aal_group['{:s}/{:s}/{:s}/{:d}/{:d}'.format(
                        space, resolution, metric, nearest_k, depth)])

            if sparse == 'sparse':
                shape = np.shape(adj_matrix)
                adj_sparse = matrix_to_sparse(adj_matrix)['sparse_matrix']
                indices = np.array(
                    [list(item[0]) for item in adj_sparse.items()])
                values = np.array([item[1] for item in adj_sparse.items()])
                return indices, values, shape, adj_matrix
            return adj_matrix
        elif metric == 'Adjacent':
            adj_matrix = np.array(
                self.aal_group['{:s}/{:s}/{:s}/adj_matrix'.format(
                    space, resolution, metric)])
            adj_matrix = adj_matrix[:ROI_num, :ROI_num]
            return adj_matrix


def calculate_distance(atlas: np.ndarray,
                       ROI_num: int = 116,
                       metric: str = 'Euclidean'):
    distance = None

    if metric == 'Euclidean':
        # Calculate the cluster centers for each brain region
        cluster_centers = []
        sparse_atlas = matrix_to_sparse(atlas)['sparse_matrix']
        for region_index in np.arange(ROI_num) + 1:
            voxels = filter(lambda x: x[1] == region_index,
                            sparse_atlas.items())
            voxels = np.array([np.array(c) for c, _ in voxels])
            cluster_center = np.mean(voxels, axis=0)
            cluster_centers.append(cluster_center)
            print('Center of region {:d} is {}'.format(region_index,
                                                       cluster_center))
        cluster_centers = np.array(cluster_centers)

        repeat_centers = np.repeat(a=np.expand_dims(cluster_centers, axis=1),
                                   axis=1,
                                   repeats=ROI_num)

        distance = np.sqrt(
            np.sum(np.square(repeat_centers -
                             np.transpose(repeat_centers, axes=[1, 0, 2])),
                   axis=2))
        return distance


def get_structure(ROI_num: int = 116,
                  measure: str = 'Euclidean',
                  atlas: str = 'AAL',
                  distance: np.ndarray = None,
                  nearest_k: int = 10):
    # Select nearest-k regions by distance matrix for each region,
    # and build a whole brain graph with element 1 in location (i,j)
    # indicate that there exist a connection between brain region i and j.

    if distance is None:
        atlas = AAL().aal_group['{:s}/{:s}/{:s}'.format()]
        distance = calculate_distance(
            atlas=atlas,
            ROI_num=ROI_num,
            metric=measure,
        )

    whole_brain_graph = np.zeros(shape=[ROI_num, ROI_num])
    for region_index in np.arange(ROI_num):
        # Firstly, we sort the list of distance to other regions for the chosen region
        distance_to_ROI = sorted(matrix_to_sparse(
            distance[region_index, :])['sparse_matrix'].items(),
                                 key=lambda x: x[1])

        # Secondly, we select the nearest-k edges connected with chosen region,
        # then set the corresponding element in the whole brain graph to 1.
        nearest_k_list = [t[0] for t in distance_to_ROI[:nearest_k]]
        nearest_k_to_ROI = np.zeros(shape=[ROI_num, 1])
        nearest_k_to_ROI[nearest_k_list] = 1
        whole_brain_graph[:, region_index] = np.squeeze(nearest_k_to_ROI)
        whole_brain_graph[region_index, :] = np.squeeze(nearest_k_to_ROI)

    return whole_brain_graph


def calculate_adjacency_matrix(nearest_k_list: list = [10],
                               max_depth: int = [5],
                               metric: str = 'Euclidean'):
    aal_hdf5 = AAL().aal_group
    metric_group = aal_hdf5['MNI/61_73_61'].require_group(
        '{:s}'.format(metric))
    distance = np.array(metric_group['distance'])
    # create_dataset_hdf5(group=metric_group,
    #                     name='distance',
    #                     data=distance)

    v_num = 90
    for depth, nearest_k in zip(max_depth, nearest_k_list):
        structure = get_structure(distance=distance[:v_num, :v_num],
                                  nearest_k=nearest_k)
        nearest_k_group = metric_group.require_group(str(nearest_k))
        create_dataset_hdf5(group=nearest_k_group,
                            name='adj_matrix',
                            data=structure)

        g = graph.Graph(adj_matrix=structure, v_num=90)
        # g.diffuse(max_depth=5)
        # sum_items = None
        for l in np.arange(start=1, stop=depth):
            adj_matrix = g.adj_by_depth(l)
            create_dataset_hdf5(group=nearest_k_group,
                                name=str(l),
                                data=adj_matrix)

            # sum_items, graph_latex = latex.graph_matmul(graph=current_graph,
            #                                             adj_matrix=adj_matrix,
            #                                             sum_items=sum_items)
            # current_graph = math.graph_conv(current_graph=current_graph,
            #                                 adj_matrix=adj_matrix)

            # print('$${:s}$$'.format(graph_latex))
        # print(current_graph)


def onehot2vector(data, class_num=2):
    data_tmp = np.zeros(shape=[np.size(data, 0)], dtype=int)
    for class_index in range(class_num):
        data_tmp[np.where(data[:, class_index] == 1)] = class_index
    return data_tmp


def preprocess_rois_aal_file_name():
    path = 'G:\Data\ABIDE\ABIDE_Cyberduck\DPARSF\\temp'
    file_names = os.listdir(path)
    for file_name in file_names:
        names = file_name.split('_')
        old_name = os.path.join(path, file_name)
        new_name = os.path.join(path, '_'.join(names[-3:]))
        os.rename(old_name, new_name)


def load_nii(file_path: str):
    img = nib.load(filename=file_path).get_fdata()
    return img


def cal_ROISignals(func_preproc: np.ndarray, atlas: np.ndarray):
    # Get shape
    [height, width, depth, time_length] = np.shape(func_preproc)
    roi_num = int(np.max(atlas))

    # Reshape to vector or matrix
    func_preproc = np.reshape(func_preproc, [-1, time_length])
    atlas = np.reshape(atlas, [-1])

    signals = np.zeros(shape=[time_length, roi_num])
    for roi_index in range(roi_num):
        signals[:, roi_index] = np.mean(
            func_preproc[np.where(atlas == roi_index + 1)], axis=0)

    return signals


# preprocess_rois_aal_file_name()


def upper_triangle(data: dict or np.ndarray):
    if isinstance(data, np.ndarray):
        if data.ndim == 4:
            d = np.squeeze(data, axis=-1)
        else:
            d = data
        assert d.ndim == 3

        [_, width, height] = np.shape(d)
        assert height == width, 'The input data must be square matrix but get shape {:}.'.format(
            np.shape(d))

        triu_indices = np.triu_indices(n=height, k=1)
        d = np.transpose(np.transpose(d, axes=[1, 2, 0])[triu_indices],
                         axes=[1, 0])

        return d

    transoformed_data = {}
    for key in data:
        if 'data' in key:
            d = np.squeeze(data[key], axis=-1)
            assert d.ndim == 3

            [_, width, height] = np.shape(d)
            assert height == width, '{:s} must be square matrix but get shape {:}.'.format(
                key, np.shape(d))

            triu_indices = np.triu_indices(n=height, k=1)
            d = np.transpose(np.transpose(d, axes=[1, 2, 0])[triu_indices],
                             axes=[1, 0])

            transoformed_data[key] = d
        elif 'label' in key:
            transoformed_data[key] = data[key]

    return transoformed_data


def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


# Local Connectivity Feature Extraction




def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)
    
    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
 
        if verbose:
            print(s, type(o), repr(o), file=stderr)
 
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
 
    return sizeof(o)

from collections import deque
from itertools import chain
from sys import getsizeof

def cal_memory():
    memory = 0
    scale = 1073741824

    vars = globals()
    vars.update(locals())

    for var in list(vars.keys()):
        m = eval("total_size({})".format(var)) / scale
        memory += m
        print('{:s}: {:e}GB'.format(var, m))
    print('Total: {}'.format(memory))

import psutil

def print_memory_status():
    # Print the amount of RAM used by the process in gigabytes
    print(psutil.Process(os.getpid()).memory_info().rss * 0.00000001)

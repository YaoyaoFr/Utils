'''
Author: your name
Date: 2020-06-11 08:54:58
LastEditTime: 2020-09-02 09:23:50
LastEditors: Yaoyao
Description: In User Settings Edit
FilePath: /Utils/Dataset/utils/small_world.py
'''
import numpy as np
import h5py
import sklearn.preprocessing as prep


def local_connectivity_pattern_extraction_fold(
    data_fold: dict,
    sparsity: float = 1,
    if_nonnegative: bool = False,
    if_diag_zero: bool = True,
    tag='data',
):
    for p in ['train', 'valid', 'test']:
        name = '{:s} {:s}'.format(p, tag)
        if name in data_fold:
            LCPs = local_connectivity_patterns(
                networks=data_fold[name],
                if_nonnegative=if_nonnegative,
                if_diag_zero=if_diag_zero,
                sparsity=sparsity,
            )['local connectivity patterns']
            data_fold['{:s} CPs'.format(p)] = LCPs

    return data_fold


def local_connectivity_pattern_extraction_fold2(
    data_fold: dict,
    threshold: float = 0
):
    for p in ['train', 'valid', 'test']:
        if '{:s} data'.format(p) in data_fold:
            functional_connectivity = data_fold['{:s} data'.format(p)]
            absolute_FC = np.abs(functional_connectivity)

            # Thresholding the insignificant connectivities
            functional_connectivity[np.where(absolute_FC < threshold)] = 0
            mask_FC = np.cast[int](absolute_FC >= threshold)

            [sample_size, node_num, node_num,
             channel_num] = np.shape(functional_connectivity)
            node_rows = np.split(
                absolute_FC, indices_or_sections=node_num, axis=1)
            mask_rows = np.split(mask_FC, indices_or_sections=node_num, axis=1)

            connectivity_patterns = []
            mask_connectivity_patterns = []
            node_index = 0
            for node_row, mask_row in zip(node_rows, mask_rows):
                # for node_row in node_rows:
                print('\rExtracting the local connectivity patterns of node {:d} in {:s} data'.format(
                    node_index+1, p), end='')

                plus = node_row + np.transpose(node_row, axes=[0, 2, 1, 3])
                mask = mask_row * np.transpose(mask_row, axes=[0, 2, 1, 3])
                connectivity_patterns.append(
                    functional_connectivity * plus / 2)
                mask_connectivity_patterns.append(
                    functional_connectivity * plus * mask / 2)

                node_index += 1

            print(
                '\nExtracting local connectivity patterns of {:s} data complete.'.format(p))
            connectivity_patterns = np.concatenate(
                connectivity_patterns, axis=-1)
            mask_connectivity_patterns = np.concatenate(
                mask_connectivity_patterns, axis=-1)

            data_fold['{:s} CPs'.format(p)] = connectivity_patterns
            data_fold['{:s} maskCPs'.format(p)] = mask_connectivity_patterns

    return data_fold


def local_connectivity_pattern_extraction_fold3(
    data_fold: dict,
    threshold: float = 0,
    tag='data',
):
    for p in ['train', 'valid', 'test']:
        name = '{:s} {:s}'.format(p, tag)
        if name in data_fold:
            FC = np.array(data_fold[name])
            FC_squeeze = np.squeeze(FC, axis=-1)

            # Thresholding the insignificant connectivities
            abs_FC = np.abs(FC)

            mask_FC = np.cast[int](abs_FC >= threshold)
            FC_masked = np.array(FC)
            FC_masked[np.where(mask_FC == 0)] = 0
            FC_masked_squeeze = np.squeeze(FC_masked, axis=-1)

            [sample_size, node_num, node_num,
             channel_num] = np.shape(FC)

            node_rows = np.split(
                abs_FC, indices_or_sections=node_num, axis=1)
            mask_rows = np.split(mask_FC, indices_or_sections=node_num, axis=1)

            connectivity_patterns = np.zeros(
                (sample_size, node_num, node_num, node_num))
            mask_connectivity_patterns = np.zeros(
                (sample_size, node_num, node_num, node_num))
            for node_index in range(node_num):
                print('\rExtracting the local connectivity patterns of node {:d} in {:s} data'.format(
                    node_index+1, p), end='')

                plus = abs_FC[:, node_index, ...] + \
                    np.transpose(
                        abs_FC[:, node_index, ...], axes=[0, 2, 1])
                mask = mask_FC[:, node_index, ...] * \
                    np.transpose(mask_FC[:, node_index, ...], axes=[0, 2, 1])

                connectivity_patterns[...,
                                      node_index] = FC_squeeze * plus / 2
                mask_connectivity_patterns[...,
                                           node_index] = FC_masked_squeeze * plus * mask / 2

        print(
            '\nExtracting local connectivity patterns of {:s} data complete.'.format(p))

        data_fold['{:s} CPs'.format(p)] = connectivity_patterns
        data_fold['{:s} maskCPs'.format(p)] = mask_connectivity_patterns

    return data_fold


def network_binary_by_sparsity(networks: np.ndarray,
                               sparsity: float = 0.4,
                               if_abs: bool = True,
                               if_fisher_z: bool = False,
                               if_nonnegative: bool = False,
                               if_diag_zero: bool = False,
                               ):
    networks = np.copy(networks)
    sample_size, node_num, node_num = np.shape(networks)

    if if_nonnegative:
        networks[np.where(networks < 0)] = 0

    if if_abs:
        networks = np.abs(networks)

    adjacency = np.zeros(shape=[sample_size, node_num, node_num])
    for index, network in enumerate(networks):
        if if_diag_zero:
            network[np.diag_indices(node_num)] = 0

        triu_elments = list(network[np.triu_indices(node_num, 1)])
        triu_elments.sort(reverse=True)
        threshold = triu_elments[int(np.floor(
            np.count_nonzero(triu_elments) * (sparsity) - 1))]

        adj = np.zeros(shape=[node_num, node_num])
        if threshold != 0:
            adj[np.where(network >= threshold)] = 1
        else:
            adj[np.where(network > threshold)] = 1

        adjacency[index, ...] = adj
    return adjacency


def data_normalization_fold_SM(
    data_fold: h5py.Group or dict,
    strategy: str = 'standarization',
    tag: str = 'CPs',
    node_num: int = 90,
):

    if isinstance(data_fold, h5py.Group):
        data_fold = {name: np.array(data_fold[name]) for name in data_fold}

    for node_index in range(node_num):
        print('\rData normalization for node {:}...'.format(
            node_index + 1), end='')
        fit_data = np.concatenate([data_fold[tag_tmp][..., node_index]
                                   for tag_tmp in data_fold if tag in tag_tmp], axis=0)
        sample_size = len(fit_data)
        fit_data = np.reshape(fit_data, newshape=[sample_size, -1])

        preprocessor = prep.StandardScaler().fit(fit_data)

        for data_name in data_fold:
            if tag in data_name:
                data = data_fold[data_name][..., node_index]
                shape = np.shape(data)
                data = np.reshape(data, newshape=[shape[0], -1])

                if strategy == 'standarization':
                    data = preprocessor.transform(data)
                else:
                    raise TypeError(
                        'The normalization strategy supported must in [\'standarization\']')

                data = np.reshape(data, newshape=shape)
                data_fold[data_name][..., node_index] = data

    return data_fold


def local_connectivity_patterns(networks: np.ndarray,
                                network_axes: list = None,
                                if_nonnegative: bool = False,
                                if_diag_zero: bool = True,
                                sparsity: float = 1,
                                ):
    shape = np.shape(networks)
    if len(shape) > 3:
        networks = np.squeeze(networks)
    assert len(np.shape(networks)) == 3, 'The rank of networks must equal 3 but got {:}'.format(
        len(np.shape(networks)))

    sample_size, node_num, node_num = np.shape(networks)

    adjacencies = network_binary_by_sparsity(
        networks=networks,
        sparsity=sparsity,
        if_nonnegative=if_nonnegative,
        if_diag_zero=if_diag_zero,
    )
    networks *= adjacencies
    weights = np.abs(networks) * adjacencies
    K = np.sum(adjacencies, axis=1)
    S = np.sum(weights, axis=1)
    K = np.expand_dims(K, axis=[1, 2])
    S = np.expand_dims(S, axis=[1, 2])

    adjacency_slices = np.split(
        adjacencies, indices_or_sections=node_num, axis=1)
    weight_slices = np.split(
        weights, indices_or_sections=node_num, axis=1)

    local_connectivity_patterns = np.zeros(
        shape=[sample_size, node_num, node_num, node_num])
    for node_index, (adj, weight) in enumerate(zip(adjacency_slices, weight_slices)):
        print('\rExtracting the local connectivity patterns of node {:d}'.format(
            node_index+1), end='')
        plus = (weight + np.transpose(weight, axes=[0, 2, 1])) / 2
        multiply = adj * np.transpose(adj, axes=[0, 2, 1])
        local_connectivity_pattern = np.sign(
            networks) * np.sqrt(adjacencies * plus * multiply * weights)
        
        vector = networks[..., node_index, :] * np.max(np.abs(local_connectivity_pattern), axis=-2)
        local_connectivity_pattern[..., node_index, :] = vector
        local_connectivity_pattern[..., :, node_index] = vector
        
        local_connectivity_patterns[...,
                                    node_index] = local_connectivity_pattern

    coef = S * (K - 1)
    coef_reciprocal = 1/coef
    coef_reciprocal[np.where(coef == 0)] = 0
    local_connectivity_patterns *= coef_reciprocal

    print('\nLocal connectivit patterns extraction complete.')

    return {
        'weight': weights,
        'adjacency': adjacencies,
        'local connectivity patterns': local_connectivity_patterns,
    }


def local_connectivity_patterns_old(networks: np.ndarray,
                                    network_axes: list = None,
                                    if_nonnegative: bool = False,
                                    if_diag_zero: bool = False,
                                    sparsity: float = 1,
                                    ):
    shape = np.shape(networks)
    if len(shape) > 3:
        networks = np.squeeze(networks)
    assert len(np.shape(networks) != 3), 'The rank of networks must equal 3 but got {:}'.format(
        len(np.shape(networks)))

    sample_size, node_num, node_num = np.shape(networks)

    adjacencies = network_binary_by_sparsity(
        networks=networks,
        sparsity=sparsity,
        if_nonnegative=if_nonnegative,
        if_diag_zero=if_diag_zero,
    )
    weights = np.abs(networks) * adjacencies
    K = np.sum(adjacencies, axis=1)
    S = np.sum(weights, axis=1)
    K = np.expand_dims(K, axis=[1, 2])
    S = np.expand_dims(S, axis=[1, 2])

    adjacency_slices = np.split(
        adjacencies, indices_or_sections=node_num, axis=1)
    weight_slices = np.split(
        weights, indices_or_sections=node_num, axis=1)

    local_connectivity_patterns = np.zeros(
        shape=[sample_size, node_num, node_num, node_num])
    for node_index, (adj, weight) in enumerate(zip(adjacency_slices, weight_slices)):
        print('\rExtracting the local connectivity patterns of node {:d}'.format(
            node_index+1), end='')
        plus = (weight + np.transpose(weight, axes=[0, 2, 1])) / 2
        multiply = adj * np.transpose(adj, axes=[0, 2, 1])
        local_connectivity_patterns[...,
                                    node_index] = adjacencies * plus * multiply

    coef = S * (K - 1)
    coef_reciprocal = 1/coef
    coef_reciprocal[np.where(coef == 0)] = 0
    local_connectivity_patterns *= coef_reciprocal

    print('\nLocal connectivit patterns extraction complete.')

    return local_connectivity_patterns


def local_clustering_coefficient_FCs(networks: np.ndarray,
                                     network_axes: list = None,
                                     if_nonnegative: bool = False,
                                     if_diag_zero: bool = False,
                                     sparsity: float = 1,
                                     ):
    results = local_connectivity_patterns(
        networks=networks,
        network_axes=network_axes,
        if_nonnegative=if_nonnegative,
        if_diag_zero=if_diag_zero,
        sparsity=sparsity,
    )
    LCPs = results['local connectivity patterns']

    LCCs = np.sum(
        LCPs,
        axis=(-3, -2),
    )
    CCs = np.mean(LCCs, axis=-1)

    return {
        'adjacency': results['adjacency'],
        'weight': results['weight'],
        'local connectivity patterns': LCPs,
        'local clustering coefficient': LCCs,
        'clustering coefficient': CCs,
    }


def local_clustering_coefficient(networks: np.ndarray,
                                 network_axes: list = None,
                                 diagnal_zero: bool = False,
                                 ):

    # The networks should be binary
    pass

    shape = list(np.shape(networks))
    if network_axes is None:
        network_axes = [-2, -1]

    other_axes = list(np.arange(len(shape) + network_axes[0]))

    assert shape[network_axes[0]] == shape[network_axes[1]
                                           ], 'The networks must be square matrices.'
    node_num = shape[network_axes[0]]

    K = np.sum(networks, axis=network_axes[0])

    network_slices = np.split(
        networks, indices_or_sections=node_num, axis=network_axes[0])
    networks_ = np.expand_dims(networks, axis=-1)

    LCPs = np.zeros(shape + [node_num])
    for node_index, adj in enumerate(network_slices):
        print('\rExtracting the local connectivity patterns of node {:d} '.format(
            node_index+1), end='')
        multiply = adj * \
            np.transpose(adj, axes=other_axes +
                         [network_axes[1], network_axes[0]])
        LCPs[..., node_index] = multiply
    LCPs *= networks_

    K = np.expand_dims(K, axis=[n-1 for n in network_axes])
    maximal_connections = K * (K - 1)
    coef = 1 / maximal_connections
    coef[np.where(K < 2)] = 0
    LCPs *= coef
    LCCs = np.sum(LCPs, axis=tuple([n-1 for n in network_axes]))
    CC = np.mean(LCCs, axis=-1)

    return {
        'Local clustering coefficient': LCCs,
        'Clustering coefficient': CC,
    }

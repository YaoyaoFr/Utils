import os

import numpy as np
import scipy.io as sio
from gcn.utils import preprocess_features
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold

from Dataset.DataBase import DataBase
from Dataset.DatasetParcellation import DatasetParcellation
from Dataset.SparseInverseCovariance import SparseInverseCovariance
from Dataset.utils.basic import (create_dataset_hdf5, data_normalization_fold,
                                 feature_selection, hdf5_handler,
                                 onehot2vector, set_diagonal_to_zero,
                                 upper_triangle, vector2onehot,
                                 )
from Dataset.utils.small_world import data_normalization_fold_SM, local_connectivity_pattern_extraction_fold
from Dataset.utils.tfrecord import write_to_tfrecord


class SchemeData():
    def __init__(
        self,
        dir_path: str = None,
        dataset_list: list = None,
    ):
        if not dir_path:
            dir_path = '/'.join(__file__.split('/')[:-5])
        self.dir_path = dir_path

        if not dataset_list:
            dataset_list = ['ABIDE_Initiative', 'ABIDE2']
        self.dataset_list = dataset_list

        self.hdf5_path = os.path.join(dir_path,
                                      'Data/SchemeData.hdf5').encode()

    def set_scheme(
        self,
        schemes: list or str,
        parcellation_name: str = None,
        features: list = None,
        reset: bool = False,
    ):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        if not parcellation_name:
            parcellation_name = {
                'ABIDE_Initiative': 'XXY parcellation',
                'ADHD200': 'ADHD200 parcellation',
                'ABIDE2': 'ABIDE2 parcellation'
            }
        if not features:
            features = ['pearson correlation Cyberduck', 'label']

        if isinstance(schemes, str):
            schemes = [schemes]

        if reset:
            self.clear_groups(schemes=schemes)

        feature_str, label_str = features

        for dataset in self.dataset_list:
            parcellation = DatasetParcellation(
                dir_path=self.dir_path).load_parcellation(
                    name=parcellation_name[dataset], cross_validation='5 fold')
            hdf5 = hdf5_handler(self.hdf5_path)
            dataset_group = hdf5.require_group(dataset)

            for scheme in schemes:
                scheme_name = 'scheme {:s}'.format(scheme)
                scheme_group = dataset_group.require_group(scheme_name)

                for fold_name in parcellation:
                    fold_group = scheme_group.require_group(fold_name)
                    fold_parcellation = parcellation[fold_name]
                    fold_data = {}
                    for p in fold_parcellation:
                        subIDs = fold_parcellation[p]
                        data = DataBase(dir_path=self.dir_path).get_data(
                            dataset_subIDs=subIDs, features=features)

                        fold_data['{:s} data'.format(p)] = np.expand_dims(
                            data[feature_str], axis=-1)
                        fold_data['{:s} label'.format(p)] = data[label_str]

                    if scheme in [
                            'BrainNetCNN', 'CNNElementWise', 'CNNGLasso', 'CNN'
                    ]:
                        fold_data = data_normalization_fold(
                            data_fold=fold_data)
                    elif 'AutoEncoder' in scheme or 'AE' in scheme:
                        fold_data = data_normalization_fold(
                            data_fold=fold_data, strategy='normalization')
                        fold_data['pretrain data'] = fold_data['train data']
                        fold_data['pretrain label'] = fold_data['train label']
                    elif 'SVM' in scheme or 'LASSO' in scheme:
                        fold_data = data_normalization_fold(
                            data_fold=fold_data,
                            strategy='normalization',
                        )
                    elif scheme in ['CNNSmallWorld', 'CNNEWHarmonic']:
                        # Calculate Adjacency matrices and degree matrices
                        for tag in ['train', 'valid', 'test']:
                            name = '{:s} data'.format(tag)
                            if name in fold_data:
                                adjacency_matrix = np.squeeze(
                                    np.abs(fold_data[name]))
                                harmonic = 1 - np.divide(
                                    adjacency_matrix,
                                    np.expand_dims(np.sum(adjacency_matrix,
                                                          axis=2),
                                                   axis=-1))
                                fold_data['{:s} harmonic'.format(
                                    tag)] = harmonic

                        # Normalization of input data
                        fold_data = data_normalization_fold(
                            data_fold=fold_data)

                        if scheme == 'CNNSmallWorld':
                            # Calculate Node-level connectivity pattern matrices
                            fold_data = local_connectivity_pattern_extraction_fold(
                                data_fold=fold_data, threshold=0.5)
                            fold_data = data_normalization_fold_SM(
                                data_fold=fold_data)

                    elif scheme == 'GCNN':
                        fold_data = {}
                        subject_IDs = DataBase(
                            dataset_list=dataset).get_dataset_subIDs()
                        num_node = len(subject_IDs)
                        data = DataBase(dir_path=self.dir_path).get_data(
                            dataset_subIDs=subject_IDs, features=features)
                        raw_features = data[feature_str]
                        fold_data['labels'] = data[label_str]

                        indexes = dict()
                        for p in fold_parcellation:
                            mask = np.zeros(shape=[
                                num_node,
                            ])
                            index = list()
                            for subID in fold_parcellation[p]:
                                mask[subject_IDs.index(subID)] = 1
                                index.append(subject_IDs.index(subID))
                            indexes[p] = index

                            fold_data['{:s} mask'.format(p)] = mask
                        for p in fold_parcellation:
                            fold_data['{:s} indexes'.format(p)] = indexes[p]

                        adjacency_matrix = np.zeros(shape=[num_node, num_node])
                        for attr in ['site', 'sex']:
                            subject_attr = [DataBase(dir_path=self.dir_path).get_attrs(dataset_subIDs=fold_parcellation[p],
                                                                                       attrs=[attr])[attr] for p in fold_parcellation]
                            subject_attr = np.concatenate(subject_attr)
                            for i in range(num_node):
                                for j in range(i+1, num_node):
                                    if subject_attr[i] == subject_attr[j]:
                                        adjacency_matrix[i, j] += 1
                                        adjacency_matrix[j, i] += 1
                        upper_features = upper_triangle(data=raw_features)
                        activation_features = np.arctanh(upper_features)

                        selected_features = feature_selection(matrix=activation_features,
                                                              labels=onehot2vector(
                                                                  fold_data['labels']),
                                                              train_ind=indexes['train'],
                                                              fnum=2000)

                        # Calculate all pairwise distances
                        distv = distance.pdist(
                            selected_features, metric='correlation')
                        # Convert to a square symmetric distance matrix
                        dist = distance.squareform(distv)
                        sigma = np.mean(dist)
                        # Get affinity from similarity matrix
                        sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
                        adjacency_matrix = adjacency_matrix * sparse_graph

                        fold_data['adjacency matrix'] = adjacency_matrix

                        preprocessedd_features = preprocess_features(
                            selected_features)
                        fold_data['features'] = preprocessedd_features

                    for name in fold_data:
                        print('Create dataset {:s}...'.format(name))
                        create_dataset_hdf5(group=fold_group,
                                            data=fold_data[name],
                                            name=name)

                    # write_to_tfrecord(dir_path='/home/ai/data/yaoyao/Data',
                    #                   dataset=dataset,
                    #                   scheme=scheme,
                    #                   fold_name=fold_name,
                    #                   data_fold=fold_data)

    def set_scheme_tfrecord(
            self,
            schemes: list or str,
            parcellation_name: str = None,
            features: list = None,
            reset: bool = False,):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        if not parcellation_name:
            parcellation_name = {
                'ABIDE_Initiative': 'XXY parcellation',
                'ADHD200': 'ADHD200 parcellation',
                'ABIDE2': 'ABIDE2 parcellation'
            }
        if not features:
            features = ['pearson correlation Cyberduck', 'label']

        if isinstance(schemes, str):
            schemes = [schemes]

        if reset:
            self.clear_groups(schemes=schemes)

        feature_str, label_str = features

        for dataset in self.dataset_list:
            parcellation = DatasetParcellation(
                dir_path=self.dir_path).load_parcellation(
                    name=parcellation_name[dataset], cross_validation='5 fold')

            for scheme in schemes:

                for fold_name in parcellation:
                    fold_parcellation = parcellation[fold_name]
                    fold_data = {}
                    for p in fold_parcellation:
                        subIDs = fold_parcellation[p]
                        data = DataBase(dir_path=self.dir_path).get_data(
                            dataset_subIDs=subIDs, features=features)

                        fold_data['{:s} data'.format(p)] = np.expand_dims(
                            data[feature_str], axis=-1)
                        fold_data['{:s} label'.format(p)] = data[label_str]

                    if scheme in ['CNNSmallWorld', 'CNNEWHarmonic']:
                        # Calculate Adjacency matrices and degree matrices
                        for tag in ['train', 'valid', 'test']:
                            name = '{:s} data'.format(tag)
                            if name in fold_data:
                                adjacency_matrix = np.squeeze(
                                    np.abs(fold_data[name]))
                                harmonic = 1 - np.divide(adjacency_matrix,
                                                         np.expand_dims(np.sum(adjacency_matrix,
                                                                               axis=2),
                                                                        axis=-1))
                                fold_data['{:s} harmonic'.format(
                                    tag)] = harmonic

                        if scheme == 'CNNSmallWorld':
                            # Calculate Node-level connectivity pattern matrices
                            print(
                                'Extracting local connectivity patterns of {:s}...'.format(fold_name))
                            fold_data = local_connectivity_pattern_extraction_fold(
                                data_fold=fold_data, sparsity=1)

                            # # Normalization
                            print(
                                'Data normalization local connectivity patterns of {:s}...'.format(fold_name))
                            fold_data = data_normalization_fold(
                                data_fold=fold_data)
                            for tag in ['CPs']:
                                fold_data = data_normalization_fold_SM( 
                                    data_fold=fold_data,
                                    tag=tag)
                            print(
                                '\nData normalization local connectivity patterns of {:s} complete'.format(fold_name))
                        elif scheme == 'CNNEWHarmonic':
                            fold_data = data_normalization_fold(
                                data_fold=fold_data)

                    elif scheme in ['CNNElementWise']:
                        fold_data = data_normalization_fold(
                            data_fold=fold_data)

                    write_to_tfrecord(dir_path='/home/ai/data/yaoyao/Data',
                                      dataset=dataset,
                                      scheme=scheme,
                                      fold_name=fold_name,
                                      data_fold=fold_data)

    def monte_calor_cross_validation(
            self,
            run_time: int,
            dataset: str = 'ABIDE',
            atlas: str = 'aal90',
            feature: str = 'pearson correlation global',
            normalization: bool = True):
        db = DataBase(dir_path=self.dir_path, dataset_list=[dataset])
        time_parce = DatasetParcellation(
            dir_path=self.dir_path).load_parcellation(
                'Monte Calor parcellation/time {:d}'.format(run_time),
                cross_validation='Monte Calor')
        dataset = {}
        for tag in time_parce:
            data = db.get_data(dataset_subIDs=time_parce[tag],
                               features=[feature, 'label'],
                               atlas=atlas)
            dataset.update({
                '{:s} data'.format(tag):
                np.expand_dims(data[feature], axis=-1),
                '{:s} label'.format(tag):
                data['label']
            })

        if normalization:
            dataset['data'] = data_normalization_fold(data_fold=dataset)

        return dataset

    def set_SIC_scheme(
        self,
        scheme: str,
        feature: str = None,
        alpha_list: list = None,
        parcellation_name: str = None,
    ):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        if not feature:
            feature = 'sparse inverse covariance XXY'
        if not alpha_list:
            alpha_list = np.arange(start=0.1, step=0.1, stop=1.01)

        parcellation = DatasetParcellation(
            dir_path=self.dir_path).load_parcellation(
                name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme {:s}'.format(scheme))

        for fold_name in parcellation:
            fold_group = scheme_group.require_group(fold_name)
            for alpha in alpha_list:
                alpha_data = {}
                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]
                    data = SparseInverseCovariance(
                        dir_path=self.dir_path).get_data(dataset_subIDs=subIDs,
                                                         alpha=alpha,
                                                         features=feature)
                    label = DataBase(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs, features=['label'])

                    if data[feature] is None:
                        break
                    else:
                        alpha_data['{:s} data'.format(p)] = data[feature]
                        alpha_data['{:s} label'.format(p)] = label['label']
                else:
                    alpha_group = fold_group.require_group(
                        '{:.2f}'.format(alpha))
                    for name in alpha_data:
                        create_dataset_hdf5(group=alpha_group,
                                            data=alpha_data[name],
                                            name=name)

    def set_DTL_scheme(self,
                       scheme: str,
                       parcellation_name: str = None,
                       features: list = None,
                       source_features: list = None,
                       normalization: bool = False):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        SchemeData.hdf5
            -group  scheme  str: 'DTLNN'
                -group  fold    str: ['fold 1', 'fold 2', ...]
                    -data   ['train data', 'valid data', 'test data',
                             'train label', 'valid label', 'test label']

        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        PARCELLATION_NAME = {
            'ABIDE_Initiative': 'XXY parcellation',
            'ABIDE2': 'ABIDE2 parcellation',
            'ADHD200': 'ADHD200 parcellation',
        }
        SOURCE_NAME = {
            'ABIDE_Initiative': ['ABIDE2', 'ADHD200'],
            'ABIDE2': ['ABIDE_Initiative', 'ADHD200'],
            'ADHD200': ['ABIDE_Initiative', 'ABIDE2'],
        }

        if not features:
            features = ['pearson correlation Cyberduck', 'label']
        if not source_features:
            source_features = ['pearson correlation RfMRIMaps', 'label']

        feature_str, label_str = features

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in self.dataset_list:
            parcellation_name = PARCELLATION_NAME[dataset]
            parcellation = DatasetParcellation(
                dir_path=self.dir_path).load_parcellation(
                    name=parcellation_name, cross_validation='5 fold')

            dataset_group = hdf5.require_group(dataset)
            scheme_group = dataset_group.require_group(
                'scheme {:s}'.format(scheme))

            for fold_name in parcellation:
                fold_group = scheme_group.require_group(fold_name)
                fold_data = {}
                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]
                    data = DataBase(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs, features=features)

                    pearson_correlation = np.expand_dims(data[feature_str],
                                                         axis=-1)
                    label = data[label_str]

                    fold_data['{:s} data'.format(p)] = pearson_correlation
                    fold_data['{:s} label'.format(p)] = label

                trian_data = fold_data['train data']
                train_label = fold_data['train label']

                # Source data

                source_feature_str, source_label_str = source_features
                source_data = DataBase(
                    dataset_list=SOURCE_NAME[dataset]).get_data(
                        features=source_features)
                source_pc = np.expand_dims(
                    source_data[source_feature_str], axis=-1)
                source_label = source_data[source_label_str]
                normal_index = np.argmax(source_label, axis=1) == 0

                fold_data['source data'] = source_pc[normal_index]
                fold_data['source label'] = source_label[normal_index]

                if normalization:
                    fold_data = data_normalization_fold(
                        strategy='normalization')

                # normal_index_train = np.argmax(train_label, axis=1) == 1
                fold_data['pretrain data'] = np.concatenate(
                    (trian_data, fold_data['source data']), axis=0)
                fold_data['pretrain label'] = np.concatenate(
                    (train_label, fold_data['source label']), axis=0)
                fold_data.pop('source data')
                fold_data.pop('source label')

                for name in fold_data:
                    create_dataset_hdf5(group=fold_group,
                                        data=fold_data[name],
                                        name=name)

    def set_gender_scheme(
        self,
        scheme: str,
        parcellation_name: str = 'XXY parcellation',
        features: list = None,
    ):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        if not parcellation_name:
            parcellation_name = 'XXY parcellation'
        if not features:
            features = ['pearson correlation XXY', 'label']

        feature_str, label_str = features

        parcellation = DatasetParcellation(
            dir_path=self.dir_path).load_parcellation(
                name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme gender {:s}'.format(scheme))

        gender_index = {'male': 0, 'female': 1}
        hdf5 = hdf5_handler(b'/home/ai/data/yaoyao/Data/DCAE_scheme.hdf5')
        for gender in ['male', 'female']:
            hdf5_gender = hdf5_handler(
                '/home/ai/data/yaoyao/Data/DCAE_scheme_{:s}.hdf5'.format(
                    gender).encode())
            gender_group = scheme_group.require_group(gender)
            for fold_name in parcellation:
                output_group = gender_group.require_group(
                    '{:s}'.format(fold_name))
                XXY_fold_group = hdf5[
                    'scheme CNNWithGLasso/ABIDE/pearson correlation/{:s}'.
                    format(fold_name)]
                fold_group = gender_group.require_group(fold_name)
                fold_data = {}
                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]

                    data = DataBase(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs, features=features)
                    label = data['label']
                    XXY_data = np.array(
                        XXY_fold_group['{:s} covariance'.format(p)])
                    XXY_label = np.array(
                        XXY_fold_group['{:s} label'.format(p)])

                    genders = DataBase(dir_path=self.dir_path).get_attrs(
                        dataset_subIDs=subIDs, attrs=['sex'])['sex']
                    gender_indexes = (
                        np.array(genders) == gender_index[gender])

                    fold_data['{:s} covariance'.format(
                        p)] = XXY_data[gender_indexes]
                    fold_data['{:s} data'.format(p)] = XXY_data[gender_indexes]
                    fold_data['{:s} label'.format(p)] = vector2onehot(
                        XXY_label[gender_indexes])

                for name in fold_data:
                    create_dataset_hdf5(group=fold_group,
                                        data=fold_data[name],
                                        name=name)

    def clear_groups(
        self,
        schemes: str or list,
        dataset_list: str or list = None,
    ):
        if dataset_list is None:
            dataset_list = self.dataset_list
        elif isinstance(dataset_list, str):
            dataset_list = [dataset_list]

        if isinstance(schemes, str):
            schemes = [schemes]

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in dataset_list:
            if dataset not in hdf5:
                continue

            dataset_group = hdf5[dataset]
            for scheme in schemes:
                scheme_name = 'scheme {:s}'.format(scheme)
                if scheme_name not in dataset_group:
                    continue

                dataset_group.pop(scheme_name)

        hdf5.close()

    def set_mcDNN_scheme(
        self,
        atlas_list: str or list = None,
        features: str or list = None,
        dataset_list: str or list = None,
        if_reset: bool = True,
    ):
        if atlas_list is None:
            atlas_list = ['aal90', 'cc200']
        elif isinstance(atlas_list, str):
            atlas_list = [atlas_list]

        if features is None:
            features = ['pearson correlation', 'label']
        elif isinstance(features, str):
            features = [features]

        if dataset_list is None:
            dataset_list = self.dataset_list
        elif isinstance(dataset_list, str):
            dataset_list = [dataset_list]

        PARCELLATION_NAME = {
            'ABIDE_Initiative': 'XXY parcellation',
            'ADHD200': 'ADHD200 parcellation',
            'ABIDE2': 'ABIDE2 parcellation'
        }
        scheme_name = 'scheme mcDNN'
        feature_str, label_str = features

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in dataset_list:
            dataset_group = hdf5.require_group(dataset)
            if if_reset and scheme_name in dataset_group:
                dataset_group.pop(scheme_name)
            scheme_group = dataset_group.require_group(scheme_name)

            parcellation_name = PARCELLATION_NAME[dataset]
            parcellation = DatasetParcellation(
                dir_path=self.dir_path).load_parcellation(
                    name=parcellation_name, cross_validation='5 fold')

            for fold_name in parcellation:
                fold_group = scheme_group.require_group(fold_name)
                fold_data = {}
                for index, atlas in enumerate(atlas_list):
                    channel_data = {}
                    for p in parcellation[fold_name]:
                        subIDs = parcellation[fold_name][p]
                        data = DataBase(dir_path=self.dir_path).get_data(
                            dataset_subIDs=subIDs,
                            atlas=atlas,
                            features=[feature_str])

                        pearson_correlation = np.expand_dims(data[feature_str],
                                                             axis=-1)
                        channel_data['{:s} data'.format(
                            p)] = pearson_correlation
                    channel_data = data_normalization_fold(channel_data)

                    for p in parcellation[fold_name]:
                        fold_data['{:s} channel{:d} data'.format(
                            p, index + 1)] = upper_triangle(
                                channel_data['{:s} data'.format(p)])

                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]
                    label = DataBase(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs,
                        atlas=atlas,
                        features=[label_str])[label_str]
                    fold_data['{:s} label'.format(p)] = label

                for name in fold_data:
                    print('Create dataset {:s}...'.format(name))
                    create_dataset_hdf5(group=fold_group,
                                        data=fold_data[name],
                                        name=name)

    def set_synthetic_scheme(self,
                             schemes: str or list,
                             if_valid: bool = True):
        if isinstance(schemes, str):
            schemes = [schemes]

        subject_num = 1096
        fold_num = 5
        train_index = None
        valid_index = None
        test_index = None

        hdf5 = hdf5_handler(self.hdf5_path)
        for num in [1, 5, 10]:
            for level in np.arange(1, 6):
                dataset = 'Syn{:}{:}'.format(num, level)
                dataset_group = hdf5.require_group(dataset)
                data_path = '/home/ai/data/yaoyao/Data/ASDSYNROIs1096/ASD{:}{:}_NETFC_SYN_Pear.mat'.format(
                    num, level)
                load_data = sio.loadmat(data_path)
                X = load_data['syn_net']
                Y = load_data['syn_all_labels']

                skf = StratifiedKFold(n_splits=fold_num)
                for fold_index, (train_index,
                                 test_index) in enumerate(skf.split(X, Y)):
                    fold_data = {}
                    fold_name = 'fold {:d}'.format(fold_index + 1)
                    X_train = [X[index] for index in train_index]
                    X_test = [X[index] for index in test_index]
                    Y_train = [Y[index] for index in train_index]
                    Y_test = np.array([Y[index] for index in test_index])
                    if if_valid:
                        skf_valid = StratifiedKFold(n_splits=fold_num - 1)
                        for train_index, valid_index in skf_valid.split(
                                X_train, Y_train):
                            X_valid = [X_train[index] for index in valid_index]
                            X_train = [X_train[index] for index in train_index]
                            Y_valid = np.array(
                                [Y[index] for index in valid_index])
                            Y_train = np.array(
                                [Y[index] for index in train_index])
                            break

                    Y_train = vector2onehot(Y_train)
                    Y_valid = vector2onehot(Y_valid)
                    Y_test = vector2onehot(Y_test)

                    X_train = np.array(X_train)
                    X_valid = np.array(X_valid)
                    X_test = np.array(X_test)
                    fold_data['train data'] = np.expand_dims(X_train, axis=-1)
                    fold_data['valid data'] = np.expand_dims(X_valid, axis=-1)
                    fold_data['test data'] = np.expand_dims(X_test, axis=-1)

                    fold_data['train label'] = Y_train
                    fold_data['valid label'] = Y_valid
                    fold_data['test label'] = Y_test

                    for scheme in schemes:
                        if scheme in ['CNN']:
                            scheme_group = dataset_group.require_group(
                                'scheme {:s}'.format(scheme))
                            fold_group = scheme_group.require_group(fold_name)

                            fold_data_tmp = dict(fold_data)
                            fold_data_tmp = data_normalization_fold(
                                data_fold=fold_data_tmp)

                            for name in fold_data_tmp:
                                print('Create dataset {:s}...'.format(name))
                                create_dataset_hdf5(group=fold_group,
                                                    data=fold_data[name],
                                                    name=name)
                        elif scheme == 'dForest':
                            save_dir = '/home/ai/data/yaoyao/Data/LJW/Synthetic/{:}{:}'.format(
                                num, level)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            save_path = '{:s}/{:s}.mat'.format(
                                save_dir, fold_name)
                            sio.savemat(save_path, fold_data)
                            print('Save data to {:s}'.format(save_path))

    def set_dForest_scheme(
        self,
        parcellation_name: str = None,
        features: list = None,
    ):
        """
        Set the scheme dataset according to parcellation from DatasetParcellation.
        :param parcellation_name:
        :param scheme:
        :param features:
        :return:
        """
        if not parcellation_name:
            parcellation_name = {
                'ABIDE_Initiative': 'XXY parcellation',
                'ADHD200': 'ADHD200 parcellation',
                'ABIDE2': 'ABIDE2 parcellation'
            }
        if not features:
            features = ['pearson correlation', 'label']

        feature_str, label_str = features

        save_dir = '/home/ai/data/yaoyao/Data/LJW/label'

        for dataset in self.dataset_list:
            dataset_dir = os.path.join(save_dir, dataset)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            parcellation = DatasetParcellation(
                dir_path=self.dir_path).load_parcellation(
                    name=parcellation_name[dataset], cross_validation='5 fold')
            dataset_data = {}
            for fold_name in parcellation:
                fold_parcellation = parcellation[fold_name]
                fold_data = {}
                for p in fold_parcellation:
                    subIDs = fold_parcellation[p]
                    data = DataBase(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs, features=features)

                    # fold_data['{:s} net'.format(p)] = data[feature_str]
                    fold_data['{:s} phenotype'.format(p)] = data[label_str]

                save_path = os.path.join(dataset_dir,
                                         '{:s} label.mat'.format(fold_name))
                sio.savemat(save_path, fold_data)
                print('Save dataset into {:s}'.format(save_path))


if __name__ == '__main__':
    dataset_list = [
        'ABIDE_Initiative',
        # 'ABIDE2',
        # 'ADHD200',
    ]
    schemes = [
        # 'RFESVM',
        # 'LASSO',
        # 'SSAE',
        # 'DTLNN',
        # 'GCNN',
        # 'CNNElementWise',
        # 'BrainNetCNN',
        # 'CNN',
        'CNNSmallWorld',
        # 'CNNEWHarmonic',
    ]

    sd = SchemeData(dataset_list=dataset_list)
    sd.set_scheme_tfrecord(schemes=schemes)
    # sd.set_scheme(schemes=schemes)
    # sd.set_DTL_scheme(scheme='DTLNN')

import os
import numpy as np
import scipy.io as sio

from Dataset.utils.basic import hdf5_handler, create_dataset_hdf5, data_normalization_fold, set_diagonal_to_zero, data_normalization
from Dataset.DatasetParcellation import DatasetParcellation
from Dataset.DataBase import DataBase
from Dataset.SparseInverseCovariance import SparseInverseCovariance

from sklearn.model_selection import KFold


class SchemeData():

    def __init__(self,
                 dir_path: str = None,
                 dataset_list: list = None,
                 ):
        if not dir_path:
            dir_path = '/'.join(__file__.split('/')[:-5])
        self.dir_path = dir_path

        if not dataset_list:
            dataset_list = ['ABIDE', 'ABIDE2']
        self.dataset_list = dataset_list
        
        self.hdf5_path = os.path.join(
            dir_path, 'Data/SchemeData.hdf5').encode()

    def set_scheme(self,
                   scheme: str,
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
            parcellation_name = 'XXY parcellation'
        if not features:
            features = ['pearson correlation XXY', 'label']
        
        feature_str, label_str = features

        parcellation = DatasetParcellation(dir_path=self.dir_path).load_parcellation(
            name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme {:s}'.format(scheme))

        for fold_name in parcellation:
            fold_group = scheme_group.require_group(fold_name)
            fold_data = {}
            for p in parcellation[fold_name]:
                subIDs = parcellation[fold_name][p]
                data = DataBase(dir_path=self.dir_path).get_data(
                    dataset_subIDs=subIDs, features=features)

                if 'SmallWorld' in scheme:
                    fold_data['{:s} data'.format(p)] = np.expand_dims(set_diagonal_to_zero(data[feature_str]),
                                                                      axis=-1)
                else:
                    fold_data['{:s} data'.format(p)] = np.expand_dims(
                        data[feature_str], axis=-1)

                if scheme == 'CNNGLasso':
                    fold_data['{:s} covariance'.format(p)] = fold_data['{:s} data'.format(p)]
                fold_data['{:s} label'.format(p)] = data[label_str]

            if 'AutoEncoder' in scheme or 'FCNN' in scheme:
                fold_data = data_normalization_fold(fold_data, strategy='normalization')
            elif 'SmallWorld' not in scheme:
                fold_data = data_normalization_fold(fold_data)

            for name in fold_data:
                create_dataset_hdf5(group=fold_group,
                                    data=fold_data[name],
                                    name=name)

    def monte_calor_cross_validation(self,
                                     run_time: int,
                                     dataset: str = 'ABIDE',
                                     atlas: str = 'aal90',
                                     feature: str = 'pearson correlation global',
                                     normalization: bool = True):
        db = DataBase(dir_path=self.dir_path, dataset_list=[dataset])
        time_parce = DatasetParcellation(dir_path=self.dir_path).load_parcellation('Monte Calor parcellation/time {:d}'.format(run_time),
                                                                                   cross_validation='Monte Calor')
        dataset = {}
        for tag in time_parce:
            data = db.get_data(dataset_subIDs=time_parce[tag],
                               features=[feature, 'label'],
                               atlas=atlas)
            dataset.update({'{:s} data'.format(tag): np.expand_dims(data[feature], axis=-1),
                            '{:s} label'.format(tag): data['label']})

        if normalization:
            dataset['data'] = data_normalization_fold(data_fold=dataset)

        return dataset

    def set_SIC_scheme(self,
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

        parcellation = DatasetParcellation(dir_path=self.dir_path).load_parcellation(
            name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme {:s}'.format(scheme))

        for fold_name in parcellation:
            fold_group = scheme_group.require_group(fold_name)
            for alpha in alpha_list:
                alpha_data = {}
                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]
                    data = SparseInverseCovariance(dir_path=self.dir_path).get_data(
                        dataset_subIDs=subIDs, alpha=alpha, features=feature)
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
        if not parcellation_name:
            parcellation_name = 'XXY parcellation'
        if not features:
            features = ['pearson correlation XXY', 'label']

        parcellation = DatasetParcellation(dir_path=self.dir_path).load_parcellation(
            name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme {:s}'.format(scheme))

        for fold_name in parcellation:
            fold_group = scheme_group.require_group(fold_name)
            fold_data = {}
            for p in parcellation[fold_name]:
                subIDs = parcellation[fold_name][p]
                data = DataBase(dir_path=self.dir_path).get_data(
                    dataset_subIDs=subIDs, features=features)

                pearson_correlation = np.expand_dims(
                    data['pearson correlation XXY'], axis=-1)
                label = data['label']

                fold_data['{:s} data'.format(p)] = pearson_correlation
                fold_data['{:s} label'.format(p)] = label

            if normalization:
                fold_data = data_normalization_fold(
                    fold_data, strategy='normalization')

            trian_data = fold_data['train data']
            train_label = fold_data['train label']
            valid_data = fold_data['valid data']
            valid_label = fold_data['valid label']

            normal_index_valid = np.argmax(valid_label, axis=1) == 1
            normal_index_train = np.argmax(train_label, axis=1) == 1
            fold_data['valid data'] = np.concatenate((trian_data[normal_index_train], valid_data[normal_index_valid]), 
                                                      axis=0)
            fold_data['valid label'] = np.concatenate((train_label[normal_index_train], valid_label[normal_index_valid]))
            
            for name in fold_data:
                create_dataset_hdf5(group=fold_group,
                                    data=fold_data[name],
                                    name=name)

    def set_gender_scheme(self, 
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

        parcellation = DatasetParcellation(dir_path=self.dir_path).load_parcellation(
            name=parcellation_name, cross_validation='5 fold')
        hdf5 = hdf5_handler(self.hdf5_path)
        scheme_group = hdf5.require_group('scheme gender {:s}'.format(scheme))

        gender_index = {'male': 0, 'femal': 1}
        hdf5 = hdf5_handler(b'/home/ai/data/yaoyao/Data/DCAE_scheme.hdf5')
        for gender in ['male']:
            hdf5_gender = hdf5_handler(b'/home/ai/data/yaoyao/Data/DCAE_scheme_{:s}.hdf5'.format(gender))
            gender_group = scheme_group.require_group(gender)
            for fold_name in parcellation:
                output_group = hdf5_gender.require_group()
                XXY_fold_group = hdf5['scheme CNNWithGLasso/ABIDE/pearson correlation/{:s}'.format(fold_name)]
                fold_group = gender_group.require_group(fold_name)
                fold_data = {}
                for p in parcellation[fold_name]:
                    subIDs = parcellation[fold_name][p]

                    data = DataBase(dir_path=self.dir_path).get_data(dataset_subIDs=subIDs, features=features)
                    label = data['label']
                    XXY_data = np.array(XXY_fold_group['{:s} covariance'.format(p)])
                    XXY_label = np.array(XXY_fold_group['{:s} label'.format(p)])

                    genders = DataBase(dir_path=self.dir_path).get_attrs(dataset_subIDs=subIDs, 
                                                                        attrs=['sex'])['sex']
                    gender_indexes = (np.array(genders) == gender_index[gender])

                    fold_data['{:s} covariance'.format(p)] = XXY_data[gender_indexes]
                    fold_data['{:s} data'.format(p)] = XXY_data[gender_indexes]
                    fold_data['{:s} label'.format(p)] = XXY_label[gender_indexes]

                for name in fold_data:
                    create_dataset_hdf5(group=fold_group,
                                        data=fold_data[name],
                                        name=name)

if __name__ == '__main__':
    sd = SchemeData(dataset_list=['ABIDE'])
    # sd.set_scheme(scheme='CNNGLasso')
    # sd.set_scheme(scheme='FCNN')
    # sd.set_SIC_scheme(scheme='SICSVM')
    # sd.set_DTL_scheme(scheme='DTLNN', normalization=True)
    sd.set_gender_scheme(scheme='CNNGLasso')


    # cal_SICs(dir_path=dir_path)
    # set_schemes(dir_path=dir_path)

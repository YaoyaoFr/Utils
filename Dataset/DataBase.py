import os
import collections

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

from Dataset.utils.basic import (create_dataset_hdf5, hdf5_handler, t_test,
                                 vector2onehot)
from AAL.ROI import load_sorted_rois


def load_nifti_data(data_path: str):
    img = nib.load(data_path)
    data = np.array(img.get_data())
    return data


def load_data(data_path: str, feature: str, ROI_nums: int = -1):
    if data_path.endswith('.nii') and feature in ['falff', 'reho', 'vmhc']:
        data = load_nifti_data(data_path)
    elif data_path.endswith('.mat') and 'ROISignals' in feature:
        ROI_signals = sio.loadmat(data_path)['ROISignals']
        data = ROI_signals[:, :ROI_nums]
    elif data_path.endswith('.1D') and 'ROISignals' in feature:
        df = pd.read_csv(data_path, sep="\t", header=0)
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        ROIs = [
            "#" + str(y)
            for y in sorted([int(x[1:]) for x in df.keys().tolist()])
        ]
        data = np.nan_to_num(df[ROIs].values)[:, :ROI_nums]
    return data


def cal_feature(raw_data: np.ndarray, feature: str):
    if 'pearson correlation' in feature:
        corr = np.nan_to_num(np.corrcoef(raw_data.T))
        corr[np.eye(N=np.size(corr, 0)) == 1] = 1
        return corr
    elif 'sparse inverse covariance' in feature:
        pass


HEAD_DICTS = {
    'ABIDE': {
        'subjectID': 'SUB_ID',
        'label': 'DX_GROUP',
        'sex': 'SEX',
        'site': 'SITE_ID',
        'hand': 'HANDEDNESS_CATEGORY',
        'age': 'AGE_AT_SCAN',
    },
    'ABIDE_Initiative': {
        'subjectID': 'SUB_ID',
        'label': 'DX_GROUP',
        'sex': 'SEX',
        'site': 'SITE_ID',
        'hand': 'HANDEDNESS_CATEGORY',
        'age': 'AGE_AT_SCAN',
        'file_ID': 'SUB_ID',
        'full_scale_IQ': 'FIQ',
        'verbal_IQ': 'VIQ',
        'performance_IQ': 'PIQ',
    },
    'ABIDE2': {
        'subjectID': 'SUB_ID',
        'label': 'DX_GROUP',
        'sex': 'SEX',
        'site': 'SITE_ID',
        'hand': 'HANDEDNESS_CATEGORY',
        'age': 'AGE_AT_SCAN',
        'file_ID': 'FILE_ID',
        'full_scale_IQ': 'FIQ',
    },
    'ADHD200': {
        'subjectID': 'Participant ID',
        'label': 'DX',
        'sex': 'Gender',
        'site': 'Participant ID',
        'hand': 'Handedness',
        'age': 'Age',
        'file_ID': 'Participant ID',
    }
}


def get_IQ(x):
    try:
        x = int(x)
        if x == -999:
            x = 0
    except Exception:
        x = 0
    return x


FUNCTION_DICTS = {
    'ABIDE_Initiative': {
        'subjectID': lambda x: '00{:}'.format(x),
        'label': lambda x: 1 if int(x) == 1 else 0,
        'sex': lambda x: x - 1,
        'site': lambda x: x.replace('_', ' '),
        'hand': lambda x: x,
        'age': lambda x: float(x),
        'file_ID': lambda x: '00{:}'.format(x),
        'full_scale_IQ': lambda x: get_IQ(x),
        'verbal_IQ': lambda x: get_IQ(x),
        'performance_IQ': lambda x: get_IQ(x),
    },
    'ABIDE2': {
        'subjectID': lambda x: str(x),
        'label': lambda x: 1 if int(x) == 1 else 0,
        'sex': lambda x: int(x) - 1,
        'site': lambda x: x.split('-')[-1],
        'hand': lambda x: x,
        'file_ID': lambda x: x,
        'age': lambda x: float(x),
        'full_scale_IQ': lambda x: get_IQ(x),
    },
    'ADHD200': {
        'subjectID': lambda x: x.split('_')[-1],
        'label': lambda x: 1 if int(x) != 0 else 1,
        'sex': lambda x: int(x),
        'site': lambda x: '_'.join(x.split('_')[1:-1]),
        'hand': lambda x: x,
        'age': lambda x: float(x),
        'file_ID': lambda x: x,
    },
}

PATH_DICTS = {
    'ABIDE_Initiative': {
        'ROISignals Cyberduck':
            'Cyberduck/Outputs/dparsf/filt_global/rois_aal/{:s}_rois_aal.1D',
        'ROISignals RfMRIMaps':
            'RfMRIMaps/Results/ROISignals_FunImgARCWF/ROISignals_{:s}.mat'
    },
    'ABIDE2': {
        'ROISignals RfMRIMaps': 'Results/ROISignals_FunImgARCWF/ROISignals_{:s}.mat',
    },
    'ADHD200': {
        'ROISignals RfMRIMaps': 'Results/ROISignals_FunImgARCWF/ROISignals_{:s}.mat'
    },
}
'''
NeuroImageData.hdf5
    -group datasetName   str: ['ABIDE', 'ABIDE_Initiative', 'ABIDE II', 'ADHD', ...]
        -attrs 'dirPath'    str
        -group 'feature'
            -group feature  str: ['pearson correlation XXY', 'pearson correlation global', ...]
                -data p-value  
                -data t-value
        -group subjectID    str: ['50601', '50602', ...]
            -attrs 'sex'    int: 0=male, 1=female
            -attrs 'age'    float
            -attrs 'hand'   str: 'L'=left hand, 'R'=right hand
            -attrs 'site'   str
            -attrs 'label'  int: 1=patients, 0=normal control
            
            -group 'feature'
                -group atlas    str: ['aal90', 'power264']
                    -data pearson correlation XXY (only under group aal90)
                    -data pearson correlation global
                    -data pearson correlation Cyberduck
                    
                    -group sparse inverse covariance XXY
                        -data lambda str: ['0.1', '0.01', '0.001']

            -group 'data'
                -data dataType  str: ['ROISignals', 'VMHC', 'ReHo', 'fALFF']
'''


class DataBase:
    def __init__(
        self,
        dir_path: str = None,
        dataset_list: str or list = None,
    ):
        if not dir_path:
            dir_path = '/'.join(__file__.split('/')[:-5])
        self.dir_path = dir_path

        if not dataset_list:
            dataset_list = ['ABIDE_Initiative', 'ADHD200']
        elif isinstance(dataset_list, str):
            dataset_list = [dataset_list]
        self.dataset_list = dataset_list

        self.hdf5_path = os.path.join(dir_path,
                                      'Data/NeuroImageData.hdf5').encode()

    def load_phenotypes(self):
        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in self.dataset_list:
            dataset_group = hdf5.require_group(dataset)
            dataset_path = os.path.join(self.dir_path, 'Data', dataset)
            dataset_group.attrs['dirPath'] = dataset_path
            phenotype_path = os.path.join(
                dataset_path, 'RfMRIMaps_{:s}_Phenotypic.csv'.format(dataset))
            pheno = pd.read_csv(phenotype_path)
            head_dict = HEAD_DICTS[dataset]
            function_dict = FUNCTION_DICTS[dataset]

            for head in head_dict:
                pheno[head] = pheno[head_dict[head]].apply(function_dict[head])

            head_list = list(head_dict.keys())
            head_list.remove('subjectID')
            subjectIDs = pheno['subjectID']
            class_num = np.max(pheno['label']) + 1
            for subID in subjectIDs:
                subject_group = dataset_group.require_group(subID)
                subject_pheno = pheno[pheno['subjectID'] == subID].iloc[0]

                for head in head_list:
                    subject_group.attrs[head] = subject_pheno[head]
                data_group = subject_group.require_group('data')
                label = np.squeeze(
                    vector2onehot(np.reshape(subject_pheno['label'],
                                             newshape=[1, 1]),
                                  class_num=class_num))
                create_dataset_hdf5(group=data_group, name='label', data=label)

        hdf5.close()

    def load_data(self, data_types: list = None, postprocess: bool = False):
        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in self.dataset_list:
            if data_types is None:
                data_types = list(PATH_DICTS[dataset].keys())

            dataset_group = hdf5[dataset]
            dataset_path = dataset_group.attrs['dirPath']

            subject_counts = len(dataset_group)
            file_not_found_counts = 0
            zeros_counts = 0
            for subID in dataset_group:
                subject_group = dataset_group[subID]
                data_group = subject_group.require_group('data')

                for data_type in data_types:
                    try:
                        data_path = os.path.join(
                            dataset_path, PATH_DICTS[dataset][data_type].format(
                                subject_group.attrs['file_ID']))
                        data = load_data(data_path=data_path,
                                         feature=data_type)

                        # Check if all zeros
                        if np.sum(data) == 0:
                            zeros_counts += 1
                            if postprocess:
                                dataset_group.pop(subID)
                            continue

                        create_dataset_hdf5(group=data_group,
                                            data=data,
                                            name=data_type,
                                            cover=False)
                    except FileNotFoundError:
                        print(
                            'Data file {:s}/{:s}/{:s} not found.'.format(subID, dataset, data_type))
                        file_not_found_counts += 1
                        if postprocess:
                            dataset_group.pop(subID)
                        continue
                    except KeyError:
                        print(
                            'The {:s} is not included in dataset {:s}'.format(
                                data_type, dataset)
                        )
                        continue

            print(
                'Subject {:d}, zeros error {:d}, file not found {:d}.'.format(
                    subject_counts, zeros_counts, file_not_found_counts))

        hdf5.close()

    def cal_features(
            self,
            atlas_list: list = ['aal90'],
            features: list = [
                #  'pearson correlation',
                'pearson correlation RfMRIMaps',
                'pearson correlation Cyberduck'
            ]):
        atlas_index = {'aal90': np.arange(90), 'cc200': np.arange(228, 418)}

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in self.dataset_list:
            dataset_group = hdf5[dataset]
            for sub_index, subID in enumerate(dataset_group):
                subject_group = dataset_group[subID]
                feature_group = subject_group.require_group('feature')
                for atlas in atlas_list:
                    cal_dict = {
                        # 'pearson correlation':
                        # 'data/ROISignals',
                        'pearson correlation RfMRIMaps':
                        'data/ROISignals RfMRIMaps',
                        'pearson correlation Cyberduck':
                        'data/ROISignals Cyberduck',
                    }
                    atlas_group = feature_group.require_group(atlas)
                    for feature in features:
                        try:
                            raw_data = None
                            if 'ROISignals' in cal_dict[feature]:
                                raw_data = np.array(subject_group[
                                    cal_dict[feature]])[:, atlas_index[atlas]]
                            data = cal_feature(raw_data=raw_data,
                                               feature=feature)
                            print('\rProcessing the {:d}th subject...'.format(
                                sub_index + 1),
                                end='')
                            create_dataset_hdf5(group=atlas_group,
                                                data=data,
                                                name=feature,
                                                show_info=False)
                        except Exception:
                            continue
            print('\r\nThe features of dataset {:s} have completed.'.format(
                dataset))

        hdf5.close()

    def get_dataset_subIDs(self, dataset: str or list = None):
        if isinstance(dataset, str):
            datasets = [dataset]
        elif isinstance(dataset, list):
            datasets = dataset
        elif dataset is None:
            datasets = self.dataset_list

        hdf5 = hdf5_handler(self.hdf5_path)

        # Transform subID to dataset/subID
        subIDs = []
        for dataset in datasets:
            dataset_group = hdf5[dataset]
            for subID in dataset_group:
                subIDs.append('{:s}/{:s}'.format(dataset, subID))
        hdf5.close()
        return subIDs

    def get_attrs(
        self,
        dataset_subIDs: str or list = None,
        attrs: list or str = None,
        check_num: bool = True,
        join: str = None,
    ):
        hdf5 = hdf5_handler(self.hdf5_path)

        if not dataset_subIDs:
            dataset_subIDs = self.get_dataset_subIDs()
        elif isinstance(dataset_subIDs, str):
            dataset_subIDs = [dataset_subIDs]

        if check_num:
            if len(dataset_subIDs) != len(dataset_subIDs):
                raise TypeError('Couldn\'t find all subjects!')

        if attrs is None:
            attrs = ['label', 'sex', 'site', 'hand', 'age']
        elif isinstance(attrs, str):
            attrs = [attrs]

        datas = {'subject IDs': dataset_subIDs}
        for attr in attrs:
            data_attr = []
            for subject_ID in dataset_subIDs:
                try:
                    data_attr.append(hdf5[subject_ID].attrs[attr])
                except:
                    pass
            datas[attr] = data_attr

        if join:
            attr_strs = []
            for sub_index, sub_ID in enumerate(dataset_subIDs):
                attr_list = [
                    '{:}'.format(datas[attr][sub_index]) for attr in attrs
                ]
                attr_str = join.join(attr_list)
                attr_strs.append(attr_str)
            datas['attr_str'] = attr_strs

        return datas

    def get_data(
        self,
        dataset_subIDs: str or list = None,
        features: str or list = None,
        atlas: str = 'aal90',
    ):
        hdf5 = hdf5_handler(self.hdf5_path)

        if dataset_subIDs is None:
            dataset_subIDs = self.get_dataset_subIDs()
        elif isinstance(dataset_subIDs, str):
            dataset_subIDs = [dataset_subIDs]

        if features is None:
            features = ['pearson correlation']
        elif isinstance(features, str):
            features = [features]

        feature_type = {
            'pearson correlation': 'feature/{:s}'.format(atlas),
            'pearson correlation RfMRIMaps': 'feature/{:s}'.format(atlas),
            'pearson correlation Cyberduck': 'feature/{:s}'.format(atlas),
            'pearson correlation XXY': 'feature/{:s}'.format(atlas),
            'ROISignals': 'data',
            'ROISignals RfMRIMaps': 'data',
            'ROISignals Cyberduck': 'data',
            'label': 'data',
        }
        datas = {}
        for feature in features:
            data = []
            for path in dataset_subIDs:
                try:
                    sub_data = np.array(hdf5['{:s}/{:s}/{:s}'.format(
                        path, feature_type[feature], feature)])
                    data.append(np.expand_dims(sub_data, axis=0))
                except KeyError:
                    print(
                        'Load feature {:s}/{:s} of subject {:s} error.'.format(
                            atlas, feature, path))
            data = np.concatenate(data, axis=0)
            datas[feature] = data

        hdf5.close()
        return datas

    def get_hdf5_structure(self):
        hdf5 = hdf5_handler(self.hdf5_path)

        structure = {}
        for dataset in hdf5:
            dataset_group = hdf5[dataset]
            dataset_structure = {}
            for subID in dataset_group:
                subject_structure = {}
                subject_group = dataset_group[subID]
                subject_structure.update(dict(subject_group.attrs))

                # Data
                for key in ['data', 'feature']:
                    data_structure = {}
                    try:
                        group = subject_group[key]
                        for data_type in group:
                            data_structure[data_type] = list(
                                group[data_type].shape)
                    except Exception:
                        continue
                    subject_structure[key] = data_structure

                dataset_structure[subID] = subject_structure

            structure[dataset] = dataset_structure
            return structure

    def load_pearson_correlation_XXY(self):
        file_path = 'G:/Data/BrainNetCNN/ALLASD_NETFC_SG_Pear.mat'
        nets = sio.loadmat(file_path)['net']
        subIDs = [
            '00{:.0f}'.format(subID)
            for subID in sio.loadmat(file_path)['phenotype'][:, 0]
        ]

        hdf5 = hdf5_handler(self.hdf5_path)
        dataset_group = hdf5.require_group('ABIDE')
        for net, subID in zip(nets, subIDs):
            atlas_group = dataset_group.require_group(subID).require_group(
                'feature').require_group('aal90')
            create_dataset_hdf5(
                group=atlas_group,
                name='pearson correlation XXY',
                data=net,
            )

    def statistical_analysis(self,
                             features: list = None,
                             top_edges: int = None,
                             p_threshold: float = None):
        """Calculate the statisitcal p-value of each connection between patients and normal
        controls. 

        Keyword Arguments:
            features {list} -- [description] (default: {None})
        """
        if features is None:
            features = ['pearson correlation XXY']

        result = {}
        for dataset in self.dataset_list:
            # Load data and label of dataset
            data = self.get_data(
                dataset_subIDs=self.get_dataset_subIDs(dataset=dataset),
                features=features)
            label = np.array(
                self.get_attrs(
                    dataset_subIDs=self.get_dataset_subIDs(dataset=dataset),
                    attrs=['label'])['label'])
            groups = {
                'normal_controls': 1,
                'patients': 0,
            }
            group_data = {
                group: {
                    feature: data[feature][np.where(label == groups[group])]
                    for feature in features
                }
                for group in groups
            }

            hdf5 = hdf5_handler(self.hdf5_path)
            for feature in features:
                feature_group = hdf5.require_group('{:s}/feature/{:s}'.format(
                    dataset, feature))

                result = t_test(
                    normal_controls=group_data['normal_controls'][feature],
                    patients=group_data['patients'][feature])

                for result_type in result:
                    create_dataset_hdf5(group=feature_group,
                                        data=result[result_type],
                                        name=result_type,
                                        cover=False)

                if feature == 'pearson correlation XXY':
                    # Analyse the most signiticant edges
                    roi_names = load_sorted_rois()
                    p_value_sparse = collections.OrderedDict(
                        sparse.dok_matrix(np.tril(result['p_value']), -1))
                    p_value_sorted = sorted(p_value_sparse.items(),
                                            key=lambda x: x[1])

                    edge_count = 0
                    if top_edges is None:
                        top_edges = len(p_value_sorted)
                    for edge, p_value in p_value_sorted:
                        if edge_count > top_edges:
                            break

                        if p_threshold is not None and p_value > p_threshold:
                            continue

                        edge_count += 1
                        print('Edge {:d}: {:s}-{:s}\tp-value:{:f}'.format(
                            edge_count, roi_names[edge[0] + 1].roi_name,
                            roi_names[edge[1] + 1].roi_name, p_value))
            hdf5.close()


if __name__ == '__main__':
    db = DataBase(dataset_list=[
        'ABIDE_Initiative',
        'ABIDE2',
        'ADHD200',
    ])
    db.load_phenotypes()
    db.load_data(data_types=['ROISignals Cyberduck', 'ROISignals RfMRIMaps'],
                 postprocess=True)
    # db.load_pearson_correlation_XXY()
    db.cal_features(features=['pearson correlation Cyberduck'],
                    atlas_list=['aal90'])
    # db.statistical_analysis(p_threshold=0.005)

    # Debug part
    # hdf5 = hdf5_handler(db.hdf5_path)
    # hdf5.pop('ABIDE2')
    # hdf5.close()
    pass

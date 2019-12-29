import os
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio

from Dataset.utils import hdf5_handler, create_dataset_hdf5, vector2onehot


def load_nifti_data(data_path: str):
    img = nib.load(data_path)
    data = np.array(img.get_data())
    return data


def load_data(data_path: str,
              feature: str,
              ROI_nums: int = -1):
    if data_path.endswith('.nii') and feature in ['falff', 'reho', 'vmhc']:
        data = load_nifti_data(data_path)
    elif data_path.endswith('.mat') and 'ROISignals' in feature:
        ROI_signals = sio.loadmat(data_path)['ROISignals']
        data = ROI_signals[:, :ROI_nums]
    elif data_path.endswith('.1D') and 'ROISignals' in feature:
        df = pd.read_csv(data_path, sep="\t", header=0)
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
        data = np.nan_to_num(df[ROIs].values())[:, :ROI_nums]
    return data


def cal_feature(raw_data: np.ndarray,
                feature: str):
    if 'pearson correlation' in feature:
        corr = np.nan_to_num(np.corrcoef(raw_data.T))
        corr[np.eye(N=np.size(corr, 0)) == 1] = 1
        return corr
    elif 'sparse inverse covariance' in feature:
        pass

head_dicts = {
    'ABIDE':
        {'subjectID': 'SUB_ID',
         'label': 'DX_GROUP',
         'sex': 'SEX',
         'site': 'SITE_ID',
         'hand': 'HANDEDNESS_CATEGORY',
         'age': 'AGE_AT_SCAN',
         },
    'ABIDE2':
        {'subjectID': 'SUB_LIST',
         'label': 'DX_GROUP',
         'sex': 'SEX',
         'site': 'SUB_LIST',
         'hand': 'HANDEDNESS_CATEGORY',
         'age': 'AGE_AT_SCAN',
         }
}

function_dicts = {
    'ABIDE':
        {'subjectID': lambda x: '00{:}'.format(x),
         'label': lambda x: int(x) - 1,
         'sex': lambda x: x - 1,
         'site': lambda x: x.replace('_', ' '),
         'hand': lambda x: x,
         'age': lambda x: float(x)
         },
    'ABIDE2':
        {'subjectID': lambda x: x,
         'label': lambda x: int(x) - 1,
         'sex': lambda x: x - 1,
         'site': lambda x: '_'.join(x.split('-')[1].split('_')[0:2]),
         'hand': lambda x: x,
         },
}

'''
NeuroImageData.hdf5
    -group datasetName   str: ['ABIDE', 'ABIDE II', 'ADHD', ...]
        -attrs 'dirPath'    str
        -group subjectID    str: ['50601', '50602', ...]
            -attrs 'sex'    int: 0=male, 1=female
            -attrs 'age'    float
            -attrs 'hand'   str: 'L'=left hand, 'R'=right hand
            -attrs 'site'   str
            -attrs 'label'  int: 0=patients, 1=normal control
            
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

        self.hdf5_path = os.path.join(dir_path, 'Data/NeuroImageData.hdf5').encode()

    def load_phenotypes(self):
        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in self.dataset_list:
            dataset_group = hdf5.require_group(dataset)
            dataset_path = os.path.join(self.dir_path, 'Data', dataset)
            dataset_group.attrs['dirPath'] = dataset_path
            phenotype_path = os.path.join(dataset_path, 'RfMRIMaps_{:s}_Phenotypic.csv'.format(dataset))
            pheno = pd.read_csv(phenotype_path)
            head_dict = head_dicts[dataset]
            function_dict = function_dicts[dataset]

            for head in head_dict:
                pheno[head] = pheno[head_dict[head]].apply(function_dict[head])

            head_list = list(head_dict.keys())
            head_list.remove('subjectID')
            subjectIDs = pheno['subjectID']
            for subID in subjectIDs:
                subject_group = dataset_group.require_group(subID)
                subject_pheno = pheno[pheno['subjectID'] == subID].iloc[0]
                for head in head_list:
                    subject_group.attrs[head] = subject_pheno[head]
                data_group = subject_group.require_group('data')
                label = np.squeeze(vector2onehot(np.reshape(subject_pheno['label'], newshape=[1, 1])))
                create_dataset_hdf5(group=data_group,
                                    name='label',
                                    data=label)

        hdf5.close()

    def load_data(self,
                  data_types: list = None):
        data_path_dict = {
            'falff': '{:s}_RfMRI/Results/fALFF_FunImgARCW/fALFFMap_{:s}.nii',
            'vmhc': '{:s}_RfMRI/Results/VMHC_FunImgARCWFsymS/zVMHCMap_{:s}.nii',
            'reho': '{:s}_RfMRI/Results/ReHo_FunImgARCWF/ReHoMap_{:s}.nii',
            'ROISignals global': '{:s}_RfMRI/Results/ROISignals_FunImgARglobalCWF/ROISignals_{:s}.mat',
            'ROISignals': '{:s}_RfMRI/Results/ROISignals_FunImgARCWF/ROISignals_{:s}.mat',
            'ROISignals Cyberduck': '{:s}_Cyberduck/DPARSF/filt_global/rois_aal/{:s}_rois_aal.1D',
        }

        if data_types is None:
            data_types = list(data_path_dict.keys())

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in hdf5:
            dataset_group = hdf5[dataset]
            dataset_path = dataset_group.attrs['dirPath']
            for subID in dataset_group:
                subject_group = dataset_group[subID]
                data_group = subject_group.require_group('data')
                for data_type in data_types:
                    data_path = os.path.join(dataset_path, data_path_dict[data_type].format(dataset, subID))
                    try:
                        data = load_data(data_path=data_path, feature=data_type)
                        create_dataset_hdf5(group=data_group,
                                            data=data,
                                            name=data_type,
                                            cover=False)
                    except Exception:
                        continue

        hdf5.close()

    def cal_features(self,
                     atlas_list: list = ['aal90', 'cc200'],
                     features: list = ['pearson correlation',
                                       'pearson correlation global',
                                       'pearson correlation Cyberduck']):
        atlas_index = {'aal90': np.arange(90),
                       'cc200': np.arange(228, 428)}

        hdf5 = hdf5_handler(self.hdf5_path)
        for dataset in hdf5:
            dataset_group = hdf5[dataset]
            for sub_index, subID in enumerate(dataset_group):
                subject_group = dataset_group[subID]
                feature_group = subject_group.require_group('feature')
                for atlas in atlas_list:
                    cal_dict = {'pearson correlation': 'data/ROISignals',
                                'pearson correlation global': 'data/ROISignals global',
                                'pearson correlation Cyberduck': 'data/ROISignals Cyberduck',
                                }
                    atlas_group = feature_group.require_group(atlas)
                    for feature in features:
                        try:
                            raw_data = None
                            if 'ROISignals' in cal_dict[feature]:
                                raw_data = np.array(subject_group[cal_dict[feature]])[:, atlas_index[atlas]]
                            data = cal_feature(raw_data=raw_data, feature=feature)
                            print('\rProcessing the {:d}th subject...'.format(sub_index + 1),
                                  end='')
                            create_dataset_hdf5(group=atlas_group,
                                                data=data,
                                                name=feature,
                                                show_info=False)
                        except Exception:
                            continue
            print('\r\nThe features of dataset {:s} have completed.'.format(dataset))

        hdf5.close()

    def get_dataset_subIDs(self,
                           dataset: str or list = None):
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

    def get_attrs(self,
                  dataset_subIDs: str or list = None,
                  attrs: list = None,
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
            attrs = ['label',
                     'sex',
                     'site',
                     'hand',
                     'age']

        datas = {'subject IDs': dataset_subIDs}
        for attr in attrs:
            data_attr = []
            for subject_ID in dataset_subIDs:
                data_attr.append(hdf5[subject_ID].attrs[attr])
            datas[attr] = data_attr

        if join:
            attr_strs = []
            for sub_index, sub_ID in enumerate(dataset_subIDs):
                attr_list = ['{:}'.format(datas[attr][sub_index]) for attr in attrs]
                attr_str = join.join(attr_list)
                attr_strs.append(attr_str)
            datas['attr_str'] = attr_strs

        return datas

    def get_data(self,
                 dataset_subIDs: str or list = None,
                 features: str or list = None,
                 atlas: str = 'aal90',
                 ):
        hdf5 = hdf5_handler(self.hdf5_path)

        if not dataset_subIDs:
            dataset_subIDs = self.get_dataset_subIDs()
        elif isinstance(dataset_subIDs, str):
            dataset_subIDs = [dataset_subIDs]

        if not features:
            features = ['pearson correlation']
        elif isinstance(features, str):
            features = [features]

        feature_type = {'pearson correlation': 'feature/{:s}'.format(atlas),
                        'pearson correlation global': 'feature/{:s}'.format(atlas),
                        'pearson correlation Cyberduck': 'feature/{:s}'.format(atlas),
                        'pearson correlation XXY': 'feature/{:s}'.format(atlas),
                        'ROISignals': 'data',
                        'ROISignals global': 'data',
                        'ROISignals Cyberduck': 'data',
                        'label': 'data',
                        }
        datas = {}
        for feature in features:
            data = []
            for path in dataset_subIDs:
                try:
                    sub_data = np.array(hdf5['{:s}/{:s}/{:s}'.format(path, feature_type[feature], feature)])
                    data.append(np.expand_dims(sub_data, axis=0))
                except KeyError:
                    print('Subject {:s} does not have feature {:s}/{:s}'.format(path, atlas, feature))
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
                            data_structure[data_type] = list(group[data_type].shape)
                    except Exception:
                        continue
                    subject_structure[key] = data_structure

                dataset_structure[subID] = subject_structure

            structure[dataset] = dataset_structure
            return structure

    def load_pearson_correlation_XXY(self):
        file_path = 'G:/Data/BrainNetCNN/ALLASD_NETFC_SG_Pear.mat'
        nets = sio.loadmat(file_path)['net']
        subIDs = ['00{:.0f}'.format(subID) for subID in sio.loadmat(file_path)['phenotype'][:, 0]]

        hdf5 = hdf5_handler(self.hdf5_path)
        dataset_group = hdf5.require_group('ABIDE')
        for net, subID in zip(nets, subIDs):
            atlas_group = dataset_group.require_group(subID).require_group('feature').require_group('aal90')
            create_dataset_hdf5(group=atlas_group,
                                name='pearson correlation XXY',
                                data=net,
                                )

# db = DataBase(dataset_list=['ABIDE'])
# db.load_phenotypes()
# db.load_data(data_types=['ROISignals global'])
# db.load_pearson_correlation_XXY()
# db.cal_features(features=['pearson correlation global'])
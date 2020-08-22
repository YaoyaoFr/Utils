import os

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold

from Dataset.DataBase import DataBase
from Dataset.utils.basic import hdf5_handler


class DatasetParcellation:

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
            self.dir_path, 'Data/DatasetParcellation.hdf5').encode()

    def set_cross_validation(self,
                             fold_num: int = 5,
                             dataset_list: list = None,
                             attrs: list = None,
                             if_valid: bool = True,
                             ):
        if dataset_list is None:
            dataset_list = self.dataset_list

        if attrs is None:
            attrs = ['site', 'label']

        for dataset in dataset_list:
            hdf5 = hdf5_handler(self.hdf5_path)
            parcellation_group = hdf5.require_group(
                '{:s} parcellation'.format(dataset))
            parcellation_group.attrs['description'] = 'The parcellation of {:s}'.format(
                dataset)
            parcellation_group.attrs['dataset'] = dataset

            db = DataBase(dataset_list=dataset)
            subject_IDs = db.get_dataset_subIDs()
            attrs_tmp = db.get_attrs(dataset_subIDs=subject_IDs,
                                     attrs=attrs,
                                     join='_')['attr_str']

            parcellation = {}
            skf = StratifiedKFold(n_splits=fold_num)
            for fold_index, (train_index, test_index) in enumerate(skf.split(subject_IDs, attrs_tmp)):
                fold_name = 'fold {:d}'.format(fold_index + 1)
                fold_group = parcellation_group.require_group(fold_name)
                train_subIDs = [subject_IDs[index] for index in train_index]
                test_subIDs = [subject_IDs[index] for index in test_index]
                train_attrs = [attrs_tmp[index] for index in train_index]
                if if_valid:
                    skf_valid = StratifiedKFold(n_splits=fold_num - 1)
                    for train_index, valid_index in skf_valid.split(train_subIDs, train_attrs):
                        valid_subIDs = [train_subIDs[index]
                                        for index in valid_index]
                        train_subIDs = [train_subIDs[index]
                                        for index in train_index]
                        break

                parcellation = {'train': train_subIDs,
                                'valid': valid_subIDs,
                                'test': test_subIDs}

                for tag in parcellation:
                    if tag in fold_group:
                        fold_group.pop(tag)

                    fold_group[tag] = [subID.encode()
                                       for subID in parcellation[tag]]
                    print('Write subject list of {:s} into group {:s}'.format(
                        tag, fold_group.name))

    def set_monte_calor_parcellation(self,
                                     dataset_list: list = None,
                                     fold_nums: int = 5,
                                     run_times: int = 100):
        if dataset_list is None:
            dataset_list = self.dataset_list
        db = DataBase(dataset_list=dataset_list)
        hdf5 = hdf5_handler(self.hdf5_path)
        parcellation_group = hdf5.require_group('Monte Calor parcellation')
        parcellation_group.attrs['dataset'] = ','.join(dataset_list)

        dataset_subIDs = db.get_dataset_subIDs(dataset=dataset_list)
        datas = db.get_attrs(dataset_subIDs=dataset_subIDs,
                             attrs=['site', 'label'],
                             join='_')

        skf_test = StratifiedKFold(n_splits=fold_nums, shuffle=True)
        skf_valid = StratifiedKFold(n_splits=fold_nums - 1, shuffle=True)
        for run_time in range(run_times):
            if 'time {:d}'.format(run_time+1) in parcellation_group:
                parcellation_group.pop('time {:d}'.format(run_time + 1))

            time_group = parcellation_group.require_group(
                'time {:d}'.format(run_time + 1))
            for (train_index, test_index) in skf_test.split(datas['subject IDs'], datas['attr_str']):
                train_valid_subjects = np.array(
                    datas['subject IDs'])[train_index]
                train_valid_attr = np.array(datas['attr_str'])[train_index]
                for (train_index, valid_index) in skf_valid.split(train_valid_subjects,
                                                                  train_valid_attr):
                    time_group['train'] = [str.encode()
                                           for str in train_valid_subjects[train_index]]
                    time_group['valid'] = [str.encode()
                                           for str in train_valid_subjects[valid_index]]
                    time_group['test'] = [str.encode() for str in np.array(
                        datas['subject IDs'])[test_index]]
                    break
                break

    def set_XXY_parcellation(self):
        dataset = 'ABIDE_Initiative'
        hdf5 = hdf5_handler(self.hdf5_path)
        parcellation_group = hdf5.require_group('XXY parcellation')
        parcellation_group.attrs['description'] = 'The parcellation of XXY'
        parcellation_group.attrs['dataset'] = dataset

        XXY_parcellation_dir = 'F:\OneDrive\Data\Data\BrainNetCNN'
        for fold_num in np.arange(5):
            fold_group = parcellation_group.require_group(
                'fold {:d}'.format(fold_num + 1))
            parcellation_path = os.path.join(
                XXY_parcellation_dir, 'ALLASD{:d}_NETFC_SG_Pear.mat'.format(fold_num + 1))
            parcellation_data = sio.loadmat(parcellation_path)

            for p in ['train', 'valid', 'test']:
                phenotype = parcellation_data['phenotype_{:s}'.format(p)]
                subject_IDs = [
                    '{:s}/00{:.0f}'.format(dataset, subID).encode() for subID in phenotype[:, 0]]
                if p in fold_group:
                    fold_group.pop(p)
                fold_group[p] = subject_IDs
                pass

    def set_ADHD200_parcellation(self,
                                 cover: bool = True,
                                 fold_num: int = 5):
        dataset = 'ADHD200'

        head_dict = {
            'subjectID': 'Participant ID',
        }

        function_dict = {
            'subjectID': lambda x: x.split('_')[-1],
        }

        test_function_dict = {
            'subjectID': lambda x: '00{:}'.format(x) if len(str(x)) < 7 else str(x),
        }

        phenotype_all_path = os.path.join(
            self.dir_path,
            'Data/{:s}/RfMRIMaps_{:s}_Phenotypic.csv'.format(dataset, dataset))
        pheno = pd.read_csv(phenotype_all_path)
        for head in head_dict:
            pheno[head] = pheno[head_dict[head]].apply(function_dict[head])
        all_subIDs = pheno['subjectID']

        phenotype_test_path = os.path .join(
            self.dir_path,
            'Data/{:s}/allSubs_testSet_phenotypic_dx.csv'.format(dataset))
        pheno = pd.read_csv(phenotype_test_path)
        for head in head_dict:
            pheno[head] = pheno[head_dict[head]].apply(
                test_function_dict[head])
        test_subIDs = list(pheno['subjectID'])

        train_subIDs = []
        for subID in all_subIDs:
            if subID not in test_subIDs:
                train_subIDs.append(subID)

        hdf5 = hdf5_handler(self.hdf5_path)
        parcellation_group = hdf5.require_group('ADHD200 parcellation')
        parcellation_group.attrs['description'] = 'The parcellation of ADHD200 according to the competition hold out test set'
        parcellation_group.attrs['dataset'] = dataset

        train_subIDs = ['ADHD200/{:s}'.format(subID) for subID in train_subIDs]
        test_subIDs = ['ADHD200/{:s}'.format(subID) for subID in test_subIDs]
        parcellation = self.set_cross_validation(fold_num=fold_num,
                                                 subject_IDs=train_subIDs,
                                                 attrs=['site', 'sex', 'label'])

        for fold_name in parcellation:
            fold_group = parcellation_group.require_group(fold_name)

            if cover:
                if 'train' in fold_group:
                    fold_group.pop('train')
                if 'valid' in fold_group:
                    fold_group.pop('valid')
                if 'test' in fold_group:
                    fold_group.pop('test')

            fold_group['train'] = [subID.encode()
                                   for subID in parcellation[fold_name]['train']]
            fold_group['valid'] = [subID.encode()
                                   for subID in parcellation[fold_name]['test']]
            fold_group['test'] = [subID.encode() for subID in test_subIDs]

    def set_ABIDE2_parcellation(self,
                                fold_num: int = 5,
                                cover: bool = True):
        dataset = 'ABIDE2'
        hdf5 = hdf5_handler(self.hdf5_path)
        parcellation_group = hdf5.require_group('ABIDE2 parcellation')
        parcellation_group.attrs['description'] = 'The parcellation of ABIDE2'
        parcellation_group.attrs['dataset'] = dataset

        db = DataBase(dataset_list=self.dataset_list)

        subject_IDs = db.get_dataset_subIDs()
        parcellation = self.set_cross_validation(fold_num=fold_num,
                                                 subject_IDs=subject_IDs,
                                                 attrs=[
                                                     'site', 'label'],
                                                 if_valid=True)

        for fold_name in parcellation:
            fold_group = parcellation_group.require_group(fold_name)

            if cover:
                if 'train' in fold_group:
                    fold_group.pop('train')
                if 'valid' in fold_group:
                    fold_group.pop('valid')
                if 'test' in fold_group:
                    fold_group.pop('test')

            fold_group['train'] = [subID.encode()
                                   for subID in parcellation[fold_name]['train']]
            fold_group['valid'] = [subID.encode()
                                   for subID in parcellation[fold_name]['valid']]
            fold_group['test'] = [subID.encode()
                                  for subID in parcellation[fold_name]['test']]

    def load_parcellation(self,
                          name: str = None,
                          cross_validation: str = '5 fold'):
        hdf5 = hdf5_handler(self.hdf5_path)
        if not name:
            name = 'XXY parcellation'
        parcellation_group = hdf5[name]

        parcellation = {}
        if 'fold' in cross_validation:
            for fold in parcellation_group:
                parcellation_fold = {}
                fold_group = parcellation_group.require_group(fold)
                for p in fold_group:
                    subIDs = [subID.decode() for subID in fold_group[p]]
                    parcellation_fold[p] = subIDs
                parcellation[fold] = parcellation_fold

        elif cross_validation == 'Monte Calor':
            for p in parcellation_group:
                subIDs = [subID.decode() for subID in parcellation_group[p]]
                parcellation[p] = subIDs

        hdf5.close()

        return parcellation

    def get_parcellation_info(self,
                              parcellation: str):
        parce = self.load_parcellation(parcellation)

        fold = parce['fold 1']
        subIDs = []
        for p in ['train', 'valid', 'test']:
            subIDs.extend(fold[p])
        hdf5_database = hdf5_handler(DataBase().hdf5_path)['ABIDE']

        site_names = ['CALTECH', 'CMU', 'KKI', 'LEUVEN', 'MAX MUN', 'NYU', 'OHSU', 'OLIN', 'PITT', 'SBL',
                      'SDSU', 'STANFORD', 'TRINITY', 'UCLA', 'UM', 'USM', 'YALE']
        count = {site_name: {0: {0: [],
                                 1: [],
                                 'age': []},
                             1: {0: [],
                                 1: [],
                                 'age': []},
                             }
                 for site_name in site_names}
        subjects = []
        for subID in subIDs:
            subject_group = hdf5_database[subID]
            for site_name in site_names:
                if site_name in subject_group.attrs['site']:
                    site = site_name

            age = subject_group.attrs['age']
            label = subject_group.attrs['label']
            sex = subject_group.attrs['sex']

            count[site][label][sex].append(age)
            count[site][label]['age'].append(age)

        subject_count = 0
        for site_name in site_names:
            print('{:s} & {:.1f} ({:.1f}) & M {:d}, F {:d} & {:.1f} ({:.1f}) & M {:d}, F {:d} \\\\'.format(
                site_name,
                np.mean(count[site_name][0]['age']),
                np.std(count[site_name][0]['age']),
                len(count[site_name][0][0]),
                len(count[site_name][0][1]),
                np.mean(count[site_name][1]['age']),
                np.std(count[site_name][1]['age']),
                len(count[site_name][1][0]),
                len(count[site_name][1][1]),
            ))
            subject_count += len(count[site_name][0][0])
            subject_count += len(count[site_name][1][0])
            subject_count += len(count[site_name][1][1])
            subject_count += len(count[site_name][0][1])

        print(subject_count)


if __name__ == '__main__':
    dp = DatasetParcellation(dataset_list=['ADHD200'])
    dp.set_cross_validation()
    # dp.set_XXY_parcellation()
    # dp.set_ADHD200_parcellation()
    # dp.set_ABIDE2_parcellation()
    # dp.set_monte_calor_parcellation()
    pass

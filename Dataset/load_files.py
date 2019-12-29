import os
import re
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.covariance as cov
import nibabel as nib

from functools import partial
from nipy.io.files import load
from sklearn.preprocessing import scale
from Dataset.utils import run_progress, format_config, compute_connectivity, create_dataset_hdf5, \
    hdf5_handler


def load_subject(subj, tmpl, feature):
    if tmpl.endswith('.nii') and feature in ['falff', 'reho', 'vmch']:
        data_path = format_config(tmpl, {
            'subject': subj,
        })
        if not os.path.exists(data_path):
            data_path = format_config(tmpl, {
                'subject': subj.split('_')[-1],
            })
        data = load_nifti_data(data_path, mask=True, normalization=True)
    # Calculate functional connectivity for .1D file
    elif tmpl.endswith('.1D'):
        df = pd.read_csv(format_config(tmpl, {
            'subject': subj,
        }), sep="\t", header=0)
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
        ROI_signals = np.nan_to_num(df[ROIs].as_matrix().T).tolist()
    elif tmpl.endswith('.mat'):
        ROI_nums = 90
        data_path = format_config(tmpl, {
            'subject': subj,
        })
        ROI_signals = sio.loadmat(data_path)['ROISignals']
        ROI_signals = ROI_signals[:, :ROI_nums]

    if feature == 'pearson correlation':
        data = np.corrcoef(ROI_signals.T)
        data = np.nan_to_num(data)
        data = compute_connectivity(functional=data)
    elif feature == 'partial correlation':
        emp_cov = cov.EmpiricalCovariance(store_precision=False)
        emp_cov.fit(ROI_signals)
        precision_matrix = emp_cov.get_precision()
        inv_variance = 1 / np.sqrt(np.repeat(np.expand_dims(np.diag(precision_matrix),
                                                            axis=-1),
                                             repeats=ROI_nums, axis=-1))
        data = precision_matrix * inv_variance * inv_variance.T
        data[np.eye(N=ROI_nums) == 1] = 1
    elif feature == 'covariance':
        data = np.cov(ROI_signals.T)
    elif feature == 'sparse inverse covariance':
        data = np.cov(ROI_signals.T)
    elif feature == 'raw_data':
        data = ROI_signals
        pass

    return subj, data


def load_patients(subjs, tmpl, feature, jobs=1):
    partial_load_patient = partial(load_subject, tmpl=tmpl, feature=feature)
    msg = 'Processing {current} of {total}'
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def load_subjects_to_file(hdf5: h5py.Group, features: list, datasets: list) -> None:
    """
    load the data of all subjects to a h5py file
    :param hdf5: the handler of h5py file for storing subjects data
    :param features: the list of feature that need to be stored
    :param datasets: the list of dataset that need to be stored
    :return: None
    """
    features_path = {
        'falff': 'fALFF_FunImgARCW/fALFFMap_{subject}.nii',
        'vmhc': 'VMHC_FunImgARCWFsymS/zVMHCMap_{subject}.nii',
        'reho': 'ReHo_FunImgARCWF/ReHoMap_{subject}.nii',
        'pearson correlation': 'ROISignals_FunImgARCWF/ROISignals_{subject}.mat',
        'partial correlation': 'ROISignals_FunImgARCWF/ROISignals_{subject}.mat',
        'covariance': 'ROISignals_FunImgARCWF/ROISignals_{subject}.mat',
        'raw_data': 'ROISignals_FunImgARCWF/ROISignals_{subject}.mat'
    }

    for dataset in datasets:
        pheno = load_phenotypes(dataset)
        download_root = 'F:/OneDriveOffL/Data/Data/{:s}/Results'.format(dataset)
        dataset_group = hdf5.require_group('{:s}/subjects'.format(dataset))
        file_ids = pheno['FILE_ID']

        for feature in features:
            file_template = os.path.join(download_root, features_path[feature])
            data = load_patients(file_ids, tmpl=file_template, feature=feature)

            for pid in data:
                record = pheno[pheno['FILE_ID'] == pid].iloc[0]
                subject_storage = dataset_group.require_group(pid)
                subject_storage.attrs['id'] = record['FILE_ID']
                subject_storage.attrs['y'] = record['DX_GROUP']
                subject_storage.attrs['site'] = record['SITE']
                subject_storage.attrs['sex'] = record['SEX']
                create_dataset_hdf5(group=subject_storage,
                                    name=feature,
                                    data=data[pid],
                                    )


def load_phenotypes(dataset: str) -> pd.DataFrame:
    if dataset == 'ABIDE':
        pheno = load_phenotypes_ABIDE_RfMRIMaps()
    elif dataset == 'ADHD':
        pheno = load_phenotypes_ADHD_RfMRIMaps()
    elif dataset == 'ABIDE II':
        pheno = load_phenotypes_ABIDE2_RfMRIMaps()
    elif dataset == 'FCP':
        pheno = load_phenotypes_FCP_RfMRIMaps()
    return pheno


def load_phenotypes_ABIDE_RfMRIMaps():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE/RfMRIMaps_ABIDE_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['SUB_ID'].apply(lambda v: '00{:}'.format(v))
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['SITE'] = pheno['SITE_ID']
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT',
                  ]]


def load_phenotypes_ABIDE2_RfMRIMaps():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE II/RfMRIMaps_ABIDE2_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['SUB_LIST']
    pheno['SITE'] = pheno['SUB_LIST'].apply(lambda v: '_'.join(v.split('-')[1].split('_')[0:2]))
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT'
                  ]]


def load_phenotypes_abide_cyberduck():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE/RfMRIMaps_ABIDE_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SITE'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'MEAN_FD',
                  'SUB_IN_SMP',
                  'STRAT']]


def load_phenotypes_ADHD_RfMRIMaps():
    pheno_path = 'F:\OneDriveOffL\Data\Data\ADHD\Phenotypic_ADHD.csv'
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['DX'] != 'pending']

    pheno['FILE_ID'] = pheno['Participant ID']
    pheno['DX_GROUP'] = pheno['DX'].apply(lambda v: 1 if int(v) > 0 else 0)
    pheno['SITE'] = pheno['Participant ID'].apply(lambda v: '_'.join(v.split('_')[1:-1]))
    pheno['SEX'] = pheno['Gender'].apply(lambda v: {1: 'F', 0: 'M', 2: 'U', }[v])
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT',
                  ]]


def load_phenotypes_FCP_RfMRIMaps():
    pheno_path = 'F:\OneDriveOffL\Data\Data\FCP\FCP_RfMRIMaps_Info.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['Subject ID'].apply(
        lambda x: ''.join(['0' for _ in np.arange(start=0, stop=3 - len(x))]) + x)
    pheno['DX_GROUP'] = pheno['DX']
    pheno['SITE'] = pheno['Site']
    pheno['SEX'] = pheno['Sex']
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT'
                  ]]


def load_subjects_data(dataset: str,
                       feature: str,
                       group: int = None,
                       hdf5: h5py.Group = None):
    if hdf5 is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
        hdf5 = hdf5_handler(hdf5_path)

    subjects_group = hdf5['{:s}/subjects'.format(dataset)]
    pheno = load_phenotypes(dataset=dataset)
    if group is None:
        subjects = list(pheno['FILE_ID'])
    else:
        subjects = list(pheno[pheno['DX_GROUP'] == group]['FILE_ID'])
    feature_data = np.array([load_subject_data(subject_group=subjects_group[subject], features=[feature])
                             for subject in subjects])
    return feature_data


def load_fold(dataset_group: h5py.Group,
              fold_group: h5py.Group,
              experiment: h5py.Group = None,
              features: list = None,
              dataset: str = None, ) -> dict:
    """
    load data in each fold given the corresponding data_group ids
    :param dataset_group: list of all data_group
    :param fold_group: The fold of cross validation
    :param experiment: The experiments settings
    :param features: the list of features to be loaded
    :param dataset: the dataset to be loaded
    :return: dictionary {'train_data' ,'valid_data', 'test_data' if exist}
    """
    if features is None:
        features = [feature.decode() for feature in experiment.attrs['features']]
    if dataset is None:
        dataset = experiment.attrs['dataset']

    datas = {}
    for flag in ['train', 'valid', 'test']:
        if flag not in fold_group:
            continue

        print('Loading  {:5s} data of {:5} in dataset: {:7s} ...'.format(flag, '_'.join(features), dataset))
        data = np.array([load_subject_data(subject_group=dataset_group[subject],
                                           features=features) for subject in fold_group[flag]])
        label = np.array([dataset_group[subject_id].attrs['y'] for subject_id in fold_group[flag]])
        datas['{:s} data'.format(flag)] = data
        datas['{:s} label'.format(flag)] = label
    return datas


def load_subject_data(subject_group: h5py.Group, features: list) -> np.ndarray:
    """
    Load data from h5py file in terms of subject list
    :param subject_group: group of subject to be loaded
    :param features: list of features to be loaded
    :return: An np.ndarray with shape of [data_shape, feature_num]
    """
    datas = []
    for feature in features:
        data = np.array(subject_group[feature])
        data = np.expand_dims(data, axis=-1)
        datas.append(data)
    datas = np.concatenate(datas, axis=-1)
    return datas


def load_nifti_data(data_path: str,
                    mask: bool = False,
                    normalization: bool = False):
    img = nib.load(data_path)
    data = np.array(img.get_data())

    atlas = None
    if mask:
        shape = np.shape(data)
        atlas_path = 'aal_{:s}.nii'.format('_'.join([str(i) for i in shape]))
        if not os.path.exists(atlas_path):
            atlas_path = 'Data/aal_{:s}.nii'.format('_'.join([str(i) for i in shape]))
        atlas = np.array(nib.load(atlas_path).get_data()) == 0
        data[atlas] = 0

    if normalization:
        mean = np.mean(data)
        var = np.var(data)
        data = (data - mean) / var
        if atlas is not None:
            data[atlas] = 0

    data = np.array(data)
    return data


import h5py
import numpy as np
from scipy import sparse

from Dataset.load_files import load_subjects_data, load_nifti_data
from Dataset.utils.basic import hdf5_handler, t_test, matrix_to_sparse, create_dataset_hdf5


def MultiInstance_patch(images, landmarks, patch_size, numofscales):
    paddingsize = patch_size
    landmarks = np.round(landmarks) + paddingsize
    landmk_num = np.size(landmarks, 0)
    batch_size = np.size(images, 0)
    patches = np.zeros((batch_size, landmk_num, patch_size, patch_size, patch_size, numofscales), dtype=float)

    while True:
        i_sample = 0

        for i_subject in range(np.size(images, 0)):
            image_subject = np.lib.pad(images[i_subject, ...], (
            (paddingsize, paddingsize), (paddingsize, paddingsize), (paddingsize, paddingsize)), 'constant',
                                       constant_values=0)
            for i_landmk in range(0, landmk_num):

                i_x = np.random.permutation(1)[0] + landmarks[i_landmk, 0]
                i_y = np.random.permutation(1)[0] + landmarks[i_landmk, 1]
                i_z = np.random.permutation(1)[0] + landmarks[i_landmk, 2]
                for i_scale in range(numofscales):
                    patches[i_sample, i_landmk, 0:patch_size, 0:patch_size, 0:patch_size, i_scale] \
                        = image_subject[int(i_x - np.floor(patch_size * (i_scale + 1) / 2)):int(
                        i_x + int(np.ceil(patch_size * (i_scale + 1) / 2.0))):i_scale + 1,
                          int(i_y - np.floor(patch_size * (i_scale + 1) / 2)):int(
                              i_y + int(np.ceil(patch_size * (i_scale + 1) / 2.0))):i_scale + 1,
                          int(i_z - np.floor(patch_size * (i_scale + 1) / 2)):int(
                              i_z + int(np.ceil(patch_size * (i_scale + 1) / 2.0))):i_scale + 1]
                    pass

            i_sample += 1

            if i_sample == batch_size:

                mi_data = []
                for j_landmk in range(0, landmk_num):
                    mi_data.append(patches[:, j_landmk, ...])

                return mi_data


def calculate_statistic(datasets: list = ['ABIDE'],
                        feature: str = 'FC',
                        hdf5: h5py.Group = None,
                        top_k_ROI: int = 5,
                        top_k_relevant: int = 1,
                        mask: dict = None,
                        ):
    """
    Select the top-k significance difference brain regions by independent two-sample t-test for functional connectivity.
    :param datasets: A list of dataset to process
    :param feature: The feature used to statistic test
    :param hdf5: The hdf5 file include all datasets
    :param top_k_ROI: The number of ROIs to be selected by t-value.
    :param top_k_relevant: The number to be selected which relevant to the ROIs selected by t-value
    :param mask:
    :return: A dict
    """
    if hdf5 is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
        hdf5 = hdf5_handler(hdf5_path)

    result = {}
    for dataset in datasets:
        groups = {'normal_controls': 0,
                  'patients': 1}

        features = {}
        for group in groups:
            feature_data = load_subjects_data(dataset=dataset,
                                              feature=feature,
                                              group=groups[group],
                                              hdf5=hdf5,
                                              )
            features[group] = feature_data

        statistic_group = hdf5['{:s}'.format(dataset)].require_group('statistic')

        if feature == 'FC':
            result = t_test(features['normal_controls'], features['patients'], mask=mask)
            ROI_num = np.shape(result['t_value'])[0]
            t_value = np.abs(result['t_value'])
            t_value = np.tril(t_value, -1)
            t_value[result['hypothesis']] = 0
            t_value_sparse = dict(sparse.dok_matrix(t_value))
            sparse_items = [item for item in t_value_sparse.items()]

            ROIs_result = [[ROI_index, 0, []] for ROI_index in range(ROI_num)]

            for t_value_tuple in sparse_items:
                coordinate, t_value = t_value_tuple
                ROIs_result[coordinate[0]][1] += t_value
                ROIs_result[coordinate[1]][1] += t_value

                ROIs_result[coordinate[0]][2].append(t_value_tuple)
                ROIs_result[coordinate[1]][2].append(((coordinate[1], coordinate[0]), t_value))

            for ROI in ROIs_result:
                ROI[2] = sorted(ROI[2], key=lambda x: x[1], reverse=True)

            # Sort and select
            ROIs_result = sorted(ROIs_result, key=lambda x: x[1], reverse=True)
            ROIs = []
            for index in range(top_k_ROI):
                ROI = ROIs_result[index]
                ROIs.append(ROI[0])
                for index_relevant in range(top_k_relevant):
                    relevant_ROI = ROI[2][index_relevant][0][1]
                    if relevant_ROI not in ROIs:
                        ROIs.append(relevant_ROI)

            result[dataset] = ROIs
        elif feature in ['reho', 'falff']:
            feature_group = statistic_group.require_group(feature)

            path = 'Data/AAL/aal_61_73_61.nii'
            aal_atlas = load_nifti_data(path)
            mask = matrix_to_sparse(aal_atlas)

            result = t_test(features['normal_controls'], features['patients'], mask=mask)

            # Sort according to t-value
            t_value = np.abs(result['t_value'])
            t_value[result['hypothesis']] = 0
            result['t_value'] = t_value

            for r in result:
                create_dataset_hdf5(group=feature_group,
                                    name=r,
                                    data=result[r],
                                    )
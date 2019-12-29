import os
import numpy as np

from Dataset.DataBase import DataBase
from Dataset.utils import hdf5_handler, create_dataset_hdf5
from sklearn.covariance import graphical_lasso

'''
SIC.hdf5
    -group datasetName   str: ['ABIDE', 'ABIDE II', 'ADHD', ...]
        -attrs 'dirPath'    str
        -group subjectID    str: ['50601', '50602', ...]
            -attrs 'sex'    int: 0=male, 1=female
            -attrs 'age'    float
            -attrs 'hand'   str: 'L'=left hand, 'R'=right hand
            -attrs 'site'   str
            -attrs 'label'  int: 0=patients, 1=normal control

            -group 'feature'
                -group atlas    str: ['aal90']
                    -group sparse inverse covariance XXY
                        -data lambda str: ['0.1', '0.01', '0.001']
'''


class SparseInverseCovariance():
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
        
        self.hdf5_path = os.path.join(dir_path, 'Data/SIC.hdf5').encode()

    def cal_SIC(self,
                alphas: list = None,
                features: list = None):
        if not alphas:
            alphas = [0.1]
        if not features:
            features = ['sparse inverse covariance XXY']

        cal_dict = {'sparse inverse covariance XXY': 'pearson correlation XXY'}
        db = DataBase(dir_path=self.dir_path,
                      dataset_list=self.dataset_list)
        dataset_subIDs = db.get_dataset_subIDs()
        raw_datas = db.get_data(dataset_subIDs,
                                features=[cal_dict[feature] for feature in features])

        hdf5 = hdf5_handler(self.hdf5_path)
        for sub_index, dataset_subID in enumerate(dataset_subIDs):
            print('*********************************************')
            atlas_group = hdf5.require_group(
                '{:s}/feature/aal90'.format(dataset_subID))

            # Feature calculation
            for feature in features:
                feature_group = atlas_group.require_group(feature)

                for alpha in alphas:
                    if '{:.2f}'.format(alpha) in feature_group:
                        continue

                    raw_data = raw_datas[cal_dict[feature]][sub_index]
                    n_features = np.size(raw_data, axis=0)
                    try:
                        covariance, precision = graphical_lasso(emp_cov=raw_data,
                                                                alpha=alpha,
                                                                mode='lars',
                                                                max_iter=200)

                        alpha_group = feature_group.require_group(
                            '{:.2f}'.format(alpha))
                        create_dataset_hdf5(group=alpha_group,
                                            name='precision',
                                            data=precision)
                        print('Subject {:s}, alpha={:f} estimated.'.format(
                            dataset_subID, alpha))
                    except Exception:
                        print('Subject {:s}, alpha={:.2f} estimation error.'.format(dataset_subID,
                                                                                    alpha))
        print('SIC estimation done.')

    def cal_BIC(self):
        db = DataBase(dir_path=self.dir_path, dataset_list=self.dataset_list)
        hdf5 = hdf5_handler(self.hdf5_path)

        dataset_subIDs = db.get_dataset_subIDs()
        for dataset_subID in dataset_subIDs:
            roi_signal = db.get_data(dataset_subIDs=dataset_subID,
                                     features='ROISignals global')['ROISignals global']
            time_length = np.size(roi_signal, axis=1)
            atlas_group = hdf5.require_group(
                '{:s}/feature/aal90'.format(dataset_subID))
            raw_data = np.array(atlas_group['pearson correlation XXY'])
            feature_group = atlas_group.require_group(
                'sparse inverse covariance XXY')

            for alpha in feature_group:
                alpha_group = feature_group.require_group(alpha)
                precision = np.array(alpha_group['precision'])

                # BIC calculation
                L = np.log(np.linalg.det(precision)) - \
                    np.sum(raw_data * precision)
                m = np.count_nonzero(precision)
                d = m * (m - 1) / 2
                BIC = -2 * L + d * np.log10(time_length)
                pass

    def get_data(self,
                 dataset_subIDs: str or list = None,
                 alpha: str or float = None,
                 features: str or list = None):
        if not dataset_subIDs:
            db = DataBase(dir_path=self.dir_path,
                          dataset_list=self.dataset_list)
            dataset_subIDs = db.get_dataset_subIDs()
        elif isinstance(dataset_subIDs, str):
            dataset_subIDs = [dataset_subIDs]

        if not alpha:
            alpha = '0.10'
        elif isinstance(alpha, float):
            alpha = '{:.2f}'.format(alpha)

        if not features:
            features = ['sparse inverse covariance XXY']
        elif isinstance(features, str):
            features = [features]

        hdf5 = hdf5_handler(self.hdf5_path)
        datas = {}
        for feature in features:
            feature_datas = []
            is_complete = True
            for dataset_subID in dataset_subIDs:
                sic_path = '{:s}/feature/aal90/{:s}/{:s}/precision'.format(
                    dataset_subID, feature, alpha)
                try:
                    data = np.expand_dims(np.array(hdf5[sic_path]), axis=0)
                    feature_datas.append(data)
                except:
                    is_complete = False
                    break
            if is_complete:
                datas[feature] = np.concatenate(feature_datas, axis=0)
            else:
                datas[feature] = None    

        return datas


def cal_SICs(dir_path: str,
             alpha_list: list = None):
    sic = SparseInverseCovariance(dir_path=dir_path, dataset_list=['ABIDE'])

    if not alpha_list:
        alpha_list = list(np.arange(start=0.1, step=0.1, stop=1.01))

    sic.cal_SIC(alphas=alpha_list)

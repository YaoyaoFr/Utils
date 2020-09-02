'''
Author: your name
Date: 2020-08-22 16:01:26
LastEditTime: 2020-08-22 16:14:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearning2.0/home/ai/data/yaoyao/Program/Python/Utils/Test/test_data_normalization_SM.py
'''
import numpy as np
from Dataset.utils.small_world import data_normalization_fold_SM

data_fold = {
    'train CPs': np.random.random((100, 10, 10, 10)),
    'valid CPs': np.random.random((50, 10, 10, 10)),
    'test CPs': np.random.random((30, 10, 10, 10))}

data_fold = data_normalization_fold_SM(data_fold=data_fold, node_num=10)

data = np.concatenate([data_fold['{:s} CPs'.format(tag)] for tag in ['train', 'valid', 'test']], axis=0)
print(np.mean(data, axis=0))
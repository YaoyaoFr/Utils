'''
Author: your name
Date: 2020-08-20 11:01:01
LastEditTime: 2020-08-23 10:37:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearning2.0/home/ai/data/yaoyao/Program/Python/Utils/Test/test_CPs.py
'''
import time
import numpy as np
from Dataset.utils.small_world import local_connectivity_pattern_extraction_fold, local_connectivity_pattern_extraction_fold2

sample_sizes = [654, 200, 220]
fold_data = {'{:s} data'.format(tag): np.random.random(
    [sample_size, 90, 90, 1]) for sample_size, tag in zip(sample_sizes, ['train', 'valid', 'test'])}

# time1 = time.clock()

# fold_data = local_connectivity_pattern_extraction_fold(
#     fold_data, threshold=0.5)

# print('Old version: {:}'.format(time2 - time1))

time2 = time.clock()
fold_data = local_connectivity_pattern_extraction_fold2(
    fold_data, threshold=0.5)

time3 = time.clock()


print('New version: {:}'.format(time3 - time2))

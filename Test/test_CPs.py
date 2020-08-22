'''
Author: your name
Date: 2020-08-20 11:01:01
LastEditTime: 2020-08-20 11:48:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearning2.0/home/ai/data/yaoyao/Program/Python/Utils/Test/test_CPs.py
'''
import numpy as np

a = np.random.random([10, 5, 5, 1])
abs_a = np.abs(a)
mask_a = np.cast[int](abs_a >= 0.5)

sample_size, node_num, node_num, channel_num = np.shape(a)

mask_rows = np.split(mask_a, indices_or_sections=node_num, axis=1)
node_rows = np.split(abs_a, indices_or_sections=node_num, axis=1)
# for node_row in node_rows:


connectivity_patterns = []
for node_row, mask_row in zip(node_rows, mask_rows):
    plus = node_row + np.transpose(node_row, axes=[0, 2, 1, 3])
    mask = mask_row * np.transpose(mask_row, axes=[0, 2, 1, 3])
    connectivity_patterns.append(plus * mask)
connectivity_patterns = np.concatenate(connectivity_patterns, axis=-1)

connectivity_patterns = np.concatenate(
    [node_row + np.transpose(node_row, axes=[0, 2, 1, 3]) * mask_row * np.transpose(mask_row, axes=[0, 2, 1, 3]) for node_row, mask_row in zip(node_rows, mask_rows)], axis=-1)
pass

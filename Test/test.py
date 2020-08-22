from Dataset.utils.basic import total_size

import numpy as np
import scipy.io as sio

from Dataset.SchemeData import SchemeData
from Dataset.utils.basic import hdf5_handler
from ops.matrix import matrix_sort

try:
    from reprlib import repr
except ImportError:
    pass
 


def test(data):
    data_b = data
    del data_b

# hdf5 = hdf5_handler(SchemeData().hdf5_path)
# data = np.array(hdf5['ABIDE_Initiative/scheme CNNSmallWorld/fold 1/train CPs'])
# print(total_size(data) / scale)
# results = matrix_sort(matrix, if_diagonal=False, top=30)
data_b = np.random.random([600, 90, 90, 90])
data = np.random.random([600, 90, 90, 90])
test(data)

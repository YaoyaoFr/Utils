import os
import numpy as np
import tensorflow as tf

'''
Author: your name
Date: 2020-08-18 11:28:16
LastEditTime: 2020-08-22 11:17:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /python/home/ai/data/yaoyao/Program/Python/Utils/Dataset/utils/tfrecord.py
'''


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


TRAIN_FEATURE = {'float': _float_feature,
                 'int': _int_feature}


SCHEME_FEATURES = {
    'CNNSmallWorld': {
        'data': {'shape': (90, 90, 1), 'type': 'float'},
        'label': {'type': 'float'},
        'CPs': {'shape': (90, 90, 90), 'type': 'float'},
        'harmonic': {'shape': (90, 90), 'type': 'float'},
        'maskCPs': {'shape': (90, 90, 90), 'type': 'float'}
    },

    'CNNElementWise': {
        'data': {'shape': (90, 90, 1), 'type': 'float'},
        'label': {'type': 'float'},
    }
}

FEATURE_DESCRIPTION = {
    'CNNElementWise': {
        'data': tf.io.FixedLenFeature([np.prod([90, 90, 1])], tf.float32),
        'data/shape': tf.io.FixedLenFeature([3], tf.int64),
        'label': tf.io.FixedLenFeature([2], tf.float32),
    },
    'CNNSmallWorld': {
        'data': tf.io.FixedLenFeature([np.prod([90, 90, 1])], tf.float32),
        'data/shape': tf.io.FixedLenFeature([3], tf.int64),
        'CPs': tf.io.FixedLenFeature([np.prod([90, 90, 90])], tf.float32),
        'CPs/shape': tf.io.FixedLenFeature([3], tf.int64),
        'harmonic': tf.io.FixedLenFeature([np.prod([90, 90])], tf.float32),
        'harmonic/shape': tf.io.FixedLenFeature([2], tf.int64),
        'label': tf.io.FixedLenFeature([2], tf.float32),
    }
}

FEATURES = {
    'CNNElementWise': ['data', 'label'],
    'CNNSmallWorld': ['data', 'CPs', 'harmonic', 'label']
}


def write_to_tfrecord(dir_path: str, dataset: str, scheme: str, fold_name: str, data_fold):
    file_path = os.path.join(dir_path, dataset, scheme, fold_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    features = SCHEME_FEATURES[scheme]
    for tag in ['train', 'valid', 'test']:
        length = len(data_fold['{:s} {:s}'.format(
            tag, list(features.keys())[0])])

        file_name = os.path.join(file_path, '{:s}.tfrecords'.format(tag))
        with tf.io.TFRecordWriter(file_name) as writer:
            for sample_index in range(length):
                print(
                    '\rProcessing {:}/{:}'.format(sample_index+1, length), end='')
                
                feature_tfrecord = {}
                for type in features.keys():
                    value = data_fold['{:s} {:s}'.format(
                        tag, type)][sample_index]
                    if 'shape' in features[type]:
                        shape = features[type]['shape']
                        feature_tfrecord['{:s}/shape'.format(type)] = TRAIN_FEATURE['int'](
                            value=list(shape))

                        reshaped_value = np.reshape(value, [-1])
                        data_feature = TRAIN_FEATURE[features[type]['type']](
                            reshaped_value)
                    else:
                        data_feature = TRAIN_FEATURE[features[type]['type']](
                            value)

                    feature_tfrecord[type] = data_feature

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature_tfrecord))
                writer.write(example.SerializeToString())
            print('\nWriting datas to {:s} complete.'.format(file_name))
            writer.close()


class TFRecordDataset(tf.data.TFRecordDataset):

    def __init__(self,
                 scheme: str,
                 fold_name: str,
                 tag: str,
                 dir_path: str = None,
                 dataset: str = None,
                 features: list = None,
                 feature_description: dict = None,
                 batch_size: int = 8,
                 ):
        if dir_path is None:
            dir_path = '/home/ai/data/yaoyao/Data'
        if dataset is None:
            dataset = 'ABIDE_Initiative'

        self.scheme = scheme
        self.fold_name = fold_name
        self.file_path = os.path.join(
            dir_path, dataset, scheme, fold_name, '{:s}.tfrecords'.format(tag))

        global features_, feature_description_
        if features is None:
            features_ = FEATURES[scheme]
        else:
            features_ = features

        if feature_description is None:
            feature_description_ = FEATURE_DESCRIPTION[scheme]
        else:
            feature_description_ = feature_description


        tf.data.TFRecordDataset.__init__(self, self.file_path)

        self.sample_nums = len([d for d in self])
        self.steps_per_epoch = int(np.ceil(self.sample_nums / batch_size))


def load_data_tfrecord(dataset: str,
                       scheme: str,
                       fold_name: str,
                       tag: str,
                       features: list = None,
                       dir_path: str = None,
                       batch_size: int = 8):
    dataset = TFRecordDataset(
        scheme=scheme, fold_name=fold_name, tag=tag, features=features)

    dataset_repeat = dataset.repeat()
    dataset_map = dataset_repeat.map(read_and_decode)
    dataset_shuffle = dataset_map.shuffle(buffer_size=100)
    dataset_batch = dataset_shuffle.batch(batch_size=batch_size)
    dataset_batch.steps_per_epoch = dataset.steps_per_epoch

    return dataset_batch


def read_and_decode(example_string):
    feature_dict = tf.io.parse_single_example(example_string,
                                              feature_description_)

    datas = []
    for feature in features_:
        data = feature_dict[feature]
        if '{:s}/shape'.format(feature) in feature_dict:
            shape = feature_dict['{:s}/shape'.format(feature)]
            data = tf.reshape(data, shape)
        datas.append(data)

    return (tuple([datas[index] for index in range(len(features_)-1)]), datas[-1])
    # return tuple(datas)

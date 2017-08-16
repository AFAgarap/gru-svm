# Copyright 2017 Abien Fred Agarap. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Makes batches of examples for training or evaluation"""
import numpy as np
from os.path import join
from os import walk
import tensorflow as tf


def file_len(filename):
    """Returns the number of lines in a file"""
    with open(filename) as file:
        for index, line in enumerate(file):
            pass
    return index + 1


def list_files(path):
    """Returns the list of files present in the path"""
    file_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        file_list.extend(join(dirpath, filename) for filename in filenames)
    return file_list


def read_from_csv(filename_queue):
    """Returns decoded CSV file in form of [0] features, [1] label, and [2] key"""

    # TF reader
    reader = tf.TextLineReader()

    # default values, in case of empty columns
    record_defaults = [[0.0] for x in range(24)]

    # returns the next record from the CSV file
    key, value = reader.read(filename_queue)

    # Get the columns from the decoded CSV file
    duration, service, src_bytes, dest_bytes, count, same_srv_rate, \
    serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count, \
    dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate, \
    flag, ids_detection, malware_detection, ashula_detection, label, src_ip_add, \
    src_port_num, dst_ip_add, dst_port_num, start_time, protocol = \
        tf.decode_csv(value, record_defaults=record_defaults)

    # group the features together
    features = tf.stack([duration, service, src_bytes, dest_bytes, count, same_srv_rate,
                         serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count,
                         dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
                         flag, ids_detection, malware_detection, ashula_detection, src_ip_add,
                         src_port_num, dst_ip_add, dst_port_num, start_time, protocol])
    # Feature Importance
    # features = tf.stack([src_bytes, same_srv_rate,
    # 					dst_host_count, dst_host_srv_count,
    # 					dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
    # 					dst_ip_add, dst_port_num, start_time, protocol])
    # features = tf.stack([dst_host_count, dst_host_srv_count, start_time])

    # return features, 1 if (label == 1) else -1, key
    return features, label


def input_pipeline(path, batch_size, num_epochs=None):
    """Batches the data from the dataset"""

    # create a list to store the filenames
    files = list_files(path=path)

    # extract filenames to a queue for input pipeline
    filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

    # gets the data from CSV
    example, label = read_from_csv(filename_queue)

    # size of buffer to be randomly sampled
    min_after_dequeue = 10 * batch_size

    # maximum amount that will be prefetched
    capacity = min_after_dequeue + 3 * batch_size

    # create batches of tensors to return
    example_batch, label_batch = tf.train.batch(
        [example, label], batch_size=batch_size, capacity=capacity)

    return example_batch, label_batch


def one_hot_encode_label(labels):
    """Returns the one-hot encoded labels"""

    # create numpy array from pandas dataframe
    labels = np.array(labels)

    # create array filled with zeros
    # shape [LENGTH, 2] for there are 2 classes
    labels_onehot = np.zeros((labels.__len__(), 2))

    # fill the 'on' bits
    labels_onehot[np.arange(labels.__len__()), labels] = 1

    # for SVM, replace 0 with -1
    # since SVM classifies y {-1, +1}
    labels_onehot[labels_onehot == 0] = -1

    return labels_onehot

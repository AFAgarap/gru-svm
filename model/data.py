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
import os
import tensorflow as tf

__version__ = '0.2'
__author__ = 'Abien Fred Agarap'


def file_len(filename):
    """Returns the number of lines in a file"""
    with open(filename) as file:
        for index, line in enumerate(file):
            pass
    return index + 1


def list_files(path):
    """Returns the list of files present in the path"""
    file_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        file_list.extend(os.path.join(dir_path, file_name) for file_name in file_names)
    return file_list


def read_from_csv(filename_queue):
    """Returns decoded CSV file in form of [0] features, [1] label, and [2] key"""

    # TF reader
    reader = tf.TextLineReader()

    # default values, in case of empty columns
    record_defaults = [[0.0] for x in range(22)]

    # returns the next record from the CSV file
    key, value = reader.read(filename_queue)

    # Get the columns from the decoded CSV file
    duration, service, src_bytes, dest_bytes, count, same_srv_rate, \
        serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count, \
        dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate, \
        flag, ids_detection, malware_detection, ashula_detection, label, \
        src_port_num, dst_port_num, start_time, protocol = \
        tf.decode_csv(value, record_defaults=record_defaults)

    # group the features together
    features = tf.stack([duration, service, src_bytes, dest_bytes, count, same_srv_rate,
                         serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count,
                         dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
                         flag, ids_detection, malware_detection, ashula_detection,
                         src_port_num, dst_port_num, start_time, protocol])

    return features, label


def input_pipeline(path, batch_size, num_classes, num_epochs=None):
    """
    Batches the data from the dataset,
    and returns one-hot encoded features and labels
    """

    # create a list to store the file names
    files = list_files(path=path)

    # extract file names to a queue for input pipeline
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)

    # gets the data from CSV
    example, label = read_from_csv(filename_queue)

    # size of buffer to be randomly sampled
    min_after_dequeue = 10 * batch_size

    # maximum amount that will be prefetched
    capacity = min_after_dequeue + 3 * batch_size

    # create batches of tensors to return
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    # one-hot encode the example_batch with depth of 10
    example_batch_onehot = tf.one_hot(tf.cast(example_batch, tf.uint8), 10, 1.0, 0.0)

    # one-hot encode the label_batch with depth of num_classes
    label_batch_onehot = tf.one_hot(tf.cast(label_batch, tf.uint8), num_classes, 1.0, -1.0)

    return example_batch_onehot, label_batch_onehot

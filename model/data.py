# Module for getting batches of preprocessed data for neural net training
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Module for data handling in the project"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.4.0'
__author__ = 'Abien Fred Agarap'

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def list_files(path):
    """Returns the list of files present in the path"""
    file_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        file_list.extend(os.path.join(dir_path, file_name) for file_name in file_names)
    return file_list


def read_from_csv(filename_queue):
    """Returns decoded CSV file in form of [0] features, [1] label"""

    # TF reader
    reader = tf.TextLineReader()

    # default values, in case of empty columns
    record_defaults = [[0.0] for _ in range(22)]

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


def input_pipeline(path, batch_size, num_classes, num_epochs):
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

    example_batch = tf.identity(example_batch, name='example_batch')  # provide a name for tensor
    label_batch = tf.identity(label_batch, name='label_batch')  # provide a name for tensor

    # one-hot encode the example_batch with depth of 10
    example_batch_onehot = tf.one_hot(tf.cast(example_batch, tf.uint8), 10, 1.0, 0.0, name='example_batch_onehot')

    # one-hot encode the label_batch with depth of num_classes
    label_batch_onehot = tf.one_hot(tf.cast(label_batch, tf.uint8), num_classes, 1.0, -1.0, name='label_batch_onehot')

    return example_batch_onehot, label_batch_onehot


def plot_accuracy(data):
    """Scatter plot the data"""
    figure = plt.figure()
    f_axes = figure.add_subplot(111)
    f_axes.scatter(data[:, 0], data[:, 1])
    plt.grid()
    plt.show()


def load_data(train_dataset, validation_dataset):
    """Loads the dataset from the specified NumPy array file"""

    # load the train data into the memory
    train_data = np.load(train_dataset)

    # load the validation data into the memory
    validation_data = np.load(validation_dataset)

    # get the train labels from the train data
    train_labels = train_data[:, 17]
    train_labels = train_labels.astype(np.float32)

    # get the train features from the train data
    train_data = np.delete(arr=train_data, obj=[17], axis=1)
    train_data = train_data.astype(np.float32)

    # get the validation labels from the validation data
    validation_labels = validation_data[:, 17]
    validation_labels = validation_labels.astype(np.float32)

    # get the validation features from the validation data
    validation_data = np.delete(arr=validation_data, obj=[17], axis=1)
    validation_data = validation_data.astype(np.float32)

    return train_data, train_labels, validation_data, validation_labels

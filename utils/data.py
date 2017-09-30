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

__version__ = '0.4.2'
__author__ = 'Abien Fred Agarap'

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(data):
    """Scatter plot the data"""
    figure = plt.figure()
    f_axes = figure.add_subplot(111)
    f_axes.scatter(data[:, 0], data[:, 1])
    plt.grid()
    plt.show()


def load_data(dataset):
    """Loads the dataset from the specified NumPy array file"""

    # load the data into memory
    data = np.load(dataset)

    # get the labels from the dataset
    labels = data[:, 17]
    labels = labels.astype(np.float32)

    # get the features from the dataset
    data = np.delete(arr=data, obj=[17], axis=1)
    data = data.astype(np.float32)

    return data, labels
